"""
Orchestrator Agent — routes queries to domain agents using Gemini tool calling.

Flow:
  1. User query → Orchestrator (Gemini with function declarations)
  2. Gemini decides which tool(s) to call (search_specs, search_crashes, etc.)
  3. Tool executes scoped Neo4j search → returns chunks
  4. Chunks sent back to Gemini as function response
  5. Gemini synthesizes final answer with citations
  6. If Gemini calls another tool (multi-hop), repeat 3-4
"""
import json
import re
from typing import Dict, List, Tuple, Optional, Callable

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_LLM_MODEL
from .agents import AGENT_REGISTRY, get_agent


# Human-readable labels for tool calls (used by UI)
TOOL_DISPLAY_NAMES = {
    "search_specs": "🔍 Searching Standard Specifications...",
    "search_construction_manual": "🔍 Searching Construction Manuals...",
    "search_materials_testing": "🔍 Searching Materials Testing Manuals...",
    "search_design_manual": "🔍 Searching Design Manuals...",
    "search_crash_data": "🔍 Searching Crash & Safety Reports...",
    "search_bridge_program": "🔍 Searching Bridge Program Documents...",
    "search_stip_planning": "🔍 Searching STIP & Planning Documents...",
    "search_admin_reports": "🔍 Searching Administrative Reports...",
    "search_general": "🔍 Searching All Documents...",
    "compare_versions": "📊 Comparing Document Versions...",
    "get_section": "📄 Retrieving Section Content...",
}

TOOL_AGENT_LABELS = {
    "search_specs": "Specs Agent",
    "search_construction_manual": "Construction Agent",
    "search_materials_testing": "Materials Agent",
    "search_design_manual": "Design Agent",
    "search_crash_data": "Safety Agent",
    "search_bridge_program": "Bridge Agent",
    "search_stip_planning": "Planning Agent",
    "search_admin_reports": "Admin Agent",
    "search_general": "General Agent",
    "compare_versions": "Specs Agent",
    "get_section": "Specs Agent",
}


# ═══════════════════════════════════════════════════════════════
#  TOOL DECLARATIONS for Gemini Function Calling
# ═══════════════════════════════════════════════════════════════

def _build_tool_declarations() -> list:
    """Build Gemini FunctionDeclaration list from agent registry."""
    declarations = []

    # ── search_<domain> tools ──
    agent_tools = {
        "search_specs": {
            "agent": "specs_agent",
            "desc": "Search Wyoming Standard Specifications for construction requirements, material specs, tolerances, thresholds, contractor obligations, penalties, warranties, insurance, change orders, disputes, certifications, environmental rules, work zone safety, concrete, asphalt, aggregate, grinding, guardrails, culverts, welding, pile driving, Section XXX references."
        },
        "search_construction_manual": {
            "agent": "construction_agent",
            "desc": "Search Construction Manuals for field inspection procedures, project administration, construction management processes."
        },
        "search_materials_testing": {
            "agent": "materials_agent",
            "desc": "Search Materials Testing Manuals for lab test procedures, sampling methods, testing frequencies, material acceptance."
        },
        "search_design_manual": {
            "agent": "design_agent",
            "desc": "Search Design Manuals for road/bridge design standards, geometric design, alignment, cross sections."
        },
        "search_crash_data": {
            "agent": "safety_agent",
            "desc": "Search Traffic Crash Reports for crash statistics, fatalities by county/year, accident trends, impaired driving data."
        },
        "search_bridge_program": {
            "agent": "bridge_agent",
            "desc": "Search Bridge Program documents for bridge design, load ratings, bridge plans, structural details."
        },
        "search_stip_planning": {
            "agent": "planning_agent",
            "desc": "Search STIP and Corridor Studies for project funding, planned projects, transportation improvement programming."
        },
        "search_admin_reports": {
            "agent": "admin_agent",
            "desc": "Search Annual Reports, Financial Reports for department accomplishments, leadership, DBE goals, organizational info."
        },
        "search_general": {
            "agent": "general_agent",
            "desc": "Search ALL documents. Use this for driver licenses, vehicle registration, permits, general WYDOT info, or when the query doesn't fit a specific domain."
        },
    }

    for tool_name, info in agent_tools.items():
        declarations.append(
            genai.protos.FunctionDeclaration(
                name=tool_name,
                description=info["desc"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "query": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The search query — what to look for in the documents.",
                        ),
                        "year": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Optional year filter (e.g. 2021, 2023). Omit to search all years.",
                        ),
                        "section": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Optional section number filter (e.g. '414', '506.4.4'). Only for specs/manuals.",
                        ),
                    },
                    required=["query"],
                ),
            )
        )

    # ── compare_versions tool ──
    declarations.append(
        genai.protos.FunctionDeclaration(
            name="compare_versions",
            description="Compare a topic between two versions of Standard Specifications (e.g., 2010 vs 2021). Returns chunks from both versions for side-by-side comparison.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "topic": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The topic to compare (e.g. 'aggregate gradation', 'portland cement', 'bridge deck requirements').",
                    ),
                    "year_old": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="The older version year (default 2010).",
                    ),
                    "year_new": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="The newer version year (default 2021).",
                    ),
                },
                required=["topic"],
            ),
        )
    )

    # ── get_section tool ──
    declarations.append(
        genai.protos.FunctionDeclaration(
            name="get_section",
            description="Get the full content of a specific numbered section from Standard Specifications (e.g., Section 414, Section 506.4.4).",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "section_number": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The section number (e.g. '414', '506.4.4', '801.1').",
                    ),
                    "year": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Optional year (default 2021).",
                    ),
                },
                required=["section_number"],
            ),
        )
    )

    return declarations


# Map tool names back to agent names
_TOOL_TO_AGENT = {
    "search_specs": "specs_agent",
    "search_construction_manual": "construction_agent",
    "search_materials_testing": "materials_agent",
    "search_design_manual": "design_agent",
    "search_crash_data": "safety_agent",
    "search_bridge_program": "bridge_agent",
    "search_stip_planning": "planning_agent",
    "search_admin_reports": "admin_agent",
    "search_general": "general_agent",
}


# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATOR SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM = """You are the WYDOT Knowledge Graph Assistant — a helpful AI that answers questions about Wyoming Department of Transportation using a knowledge graph with 1,128 documents and 88,907 text chunks.

IMPORTANT: You must ALWAYS call at least one search tool for EVERY query. Never refuse a query without searching first. If a query seems general or outside a specific domain, use search_general to search across all documents. WYDOT documents contain information about people (directors, governors), organizations, policies, budgets, and general Wyoming transportation topics.

For each user query:
1. DECIDE which tool(s) to call based on the query topic.
2. CALL the appropriate tool(s) with a clear search query. When unsure, use search_general.
3. READ the returned document chunks carefully.
4. SYNTHESIZE a comprehensive answer with citations.

CITATION RULES:
- Reference sources as [Source 1], [Source 2], etc. where the number matches the source number in the returned chunks.
- IMPORTANT: Use exactly this format: [Source X] with capital S and space before the number. Examples: [Source 1], [Source 2, Source 3].
- Include the document title, section, and year when citing.
- If information comes from multiple sources, cite all of them.
- If the retrieved chunks don't fully answer the question, share what you found and note the gaps.

MULTI-STEP REASONING:
- For comparison queries, call compare_versions or call the same tool with different year parameters.
- For cross-domain queries (e.g., "compare crash data with STIP projects"), call multiple tools.
- You can make up to 5 tool calls per query.

ANSWER FORMAT:
- Use markdown with headers, bullet points, and tables where appropriate.
- Be thorough but concise.
- Always ground answers in the retrieved document content.
"""


# ═══════════════════════════════════════════════════════════════
#  EXECUTE TOOL CALL
# ═══════════════════════════════════════════════════════════════

def _execute_tool_call(tool_name: str, args: dict) -> Tuple[str, List[Dict]]:
    """Execute a tool call and return (formatted_results, raw_chunks)."""
    query = args.get("query", "")
    year = args.get("year")
    section = args.get("section")

    print(f"  🔧 Tool: {tool_name}(query='{query[:50]}', year={year}, section={section})")

    if tool_name in _TOOL_TO_AGENT:
        agent_name = _TOOL_TO_AGENT[tool_name]
        agent = get_agent(agent_name)
        chunks = agent.search(query, year=year, section=section)
        print(f"     → {len(chunks)} chunks returned from {agent_name}")
        return agent.format_results(chunks), chunks

    elif tool_name == "compare_versions":
        topic = args.get("topic", query)
        year_old = args.get("year_old", 2010)
        year_new = args.get("year_new", 2021)
        agent = get_agent("specs_agent")
        result = agent.compare_versions(topic, year_old, year_new)

        all_chunks = result["old_version"]["chunks"] + result["new_version"]["chunks"]
        old_text = agent.format_results(result["old_version"]["chunks"])
        new_text = agent.format_results(result["new_version"]["chunks"])
        return f"=== {year_old} VERSION ===\n{old_text}\n\n=== {year_new} VERSION ===\n{new_text}", all_chunks

    elif tool_name == "get_section":
        section_num = args.get("section_number", "")
        yr = args.get("year", 2021)
        agent = get_agent("specs_agent")
        chunks = agent.get_section(section_num, year=yr)
        print(f"     → {len(chunks)} chunks from Section {section_num}")
        return agent.format_results(chunks), chunks

    return "Unknown tool.", []


# ═══════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_orchestrator(
    query: str,
    chat_history: Optional[list] = None,
    on_tool_call: Optional[Callable] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Main entry point: takes a user query, runs the orchestrator with tool calling,
    returns (answer_text, list_of_source_chunks, list_of_tool_call_events).

    Args:
        query: User's question
        chat_history: Previous conversation messages
        on_tool_call: Optional callback(tool_name, args, chunk_count) called when a tool executes

    Returns:
        answer: str — the final answer with citations
        sources: list — all chunks used to generate the answer
        tool_events: list — metadata about each tool call for UI display
    """
    genai.configure(api_key=GEMINI_API_KEY)

    tool_declarations = _build_tool_declarations()

    model = genai.GenerativeModel(
        GEMINI_LLM_MODEL,
        tools=[genai.protos.Tool(function_declarations=tool_declarations)],
        system_instruction=ORCHESTRATOR_SYSTEM,
    )

    # Build chat history if provided
    history = []
    if chat_history:
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    print(f"\n{'='*60}")
    print(f"🤖 Orchestrator received: '{query[:80]}'")
    print(f"{'='*60}")

    all_source_chunks = []
    tool_events = []  # Track all tool calls for UI

    # Send initial query
    response = chat.send_message(query)

    # Tool-calling loop (max 5 iterations)
    for iteration in range(5):
        # Check if response has function calls
        parts = response.candidates[0].content.parts

        has_function_call = any(
            hasattr(part, 'function_call') and part.function_call.name
            for part in parts
        )

        if not has_function_call:
            break  # No more tool calls — we have the final answer

        # Process ALL function calls in this response (parallel tool calling)
        function_responses = []
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call.name:
                fc = part.function_call
                tool_name = fc.name
                args = dict(fc.args)

                print(f"\n  📞 Iteration {iteration+1}: Calling {tool_name}")

                # Notify callback before execution
                if on_tool_call:
                    on_tool_call("start", tool_name, args, 0)

                result_text, chunks = _execute_tool_call(tool_name, args)
                all_source_chunks.extend(chunks)

                # Record tool event
                tool_events.append({
                    "tool_name": tool_name,
                    "agent_label": TOOL_AGENT_LABELS.get(tool_name, "Unknown"),
                    "display_name": TOOL_DISPLAY_NAMES.get(tool_name, f"Calling {tool_name}..."),
                    "args": args,
                    "chunk_count": len(chunks),
                    "iteration": iteration + 1,
                })

                # Notify callback after execution
                if on_tool_call:
                    on_tool_call("done", tool_name, args, len(chunks))

                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result_text},
                        )
                    )
                )

        # Send all function responses back
        response = chat.send_message(
            genai.protos.Content(parts=function_responses)
        )

    # Extract final text
    answer = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            answer += part.text

    print(f"\n  ✅ Answer generated ({len(answer)} chars, {len(all_source_chunks)} source chunks)")

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in all_source_chunks:
        key = s.get("id", id(s))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return answer, unique_sources, tool_events


async def run_orchestrator_async(
    query: str,
    chat_history: Optional[list] = None,
    on_tool_call: Optional[Callable] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Async wrapper for Chainlit compatibility."""
    import asyncio
    return await asyncio.to_thread(run_orchestrator, query, chat_history, on_tool_call)
