#!/usr/bin/env python3
"""
WYDOT Multi-Agent Knowledge Graph Chatbot
==========================================
Uses Gemini tool calling to route queries to specialized domain agents,
each searching a scoped subset of the Neo4j knowledge graph.

Features:
  - Live loading indicators during tool calling (shows which agent is active)
  - Side-panel source chunks (clickable, matching chatapp_gemini.py pattern)
  - Agent routing info in response header

Run:
  cd graph_processing/
  chainlit run chatapp_agentic.py -w --port 8002
"""
import os
import sys
import re
import time
import asyncio

# Add parent directory to path
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import chainlit as cl
from chainlit.input_widget import Select, Switch

from agentic_solution.orchestrator import (
    run_orchestrator_async,
    TOOL_DISPLAY_NAMES,
    TOOL_AGENT_LABELS,
)
from agentic_solution.config import GEMINI_API_KEY


# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("WYDOT Multi-Agent Knowledge Graph Chatbot")
print("=" * 60)
print(f"  GEMINI_API_KEY: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
print(f"  Mode: Multi-Agent with Tool Calling")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  WELCOME MESSAGE
# ═══════════════════════════════════════════════════════════════

WELCOME_MESSAGE = """## 🚀 WYDOT Multi-Agent Assistant

I use **specialized agents** to search across **1,128 WYDOT documents** (88,907 chunks).

Each agent searches a scoped subset of the knowledge graph for faster, more accurate results:

| Agent | Domain | Documents |
|:------|:-------|:----------|
| 🔧 **Specs Agent** | Standard Specifications | 2010, 2021 editions |
| 🏗️ **Construction Agent** | Construction Manuals | 2018–2026 |
| 🧪 **Materials Agent** | Materials Testing Manuals | Lab procedures |
| 📐 **Design Agent** | Road & Bridge Design Manuals | Design standards |
| 🚨 **Safety Agent** | Traffic Crash Reports | Crash statistics |
| 🌉 **Bridge Agent** | Bridge Design & Load Ratings | Structural details |
| 📋 **Planning Agent** | STIP & Corridor Studies | Project funding |
| 📊 **Admin Agent** | Annual & Financial Reports | Department info |
| 📁 **General Agent** | Everything else | Permits, licenses |

**Try asking:**
- *"What is the surface grinding threshold in the 2021 specs?"*
- *"Compare aggregate gradation requirements between 2010 and 2021"*
- *"How many traffic fatalities were there in Wyoming in 2022?"*
"""


# ═══════════════════════════════════════════════════════════════
#  CHAINLIT HANDLERS
# ═══════════════════════════════════════════════════════════════

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    cl.user_session.set("chat_history", [])

    settings = await cl.ChatSettings(
        [
            Switch(id="show_sources", label="Show Source Chunks", initial=True),
            Switch(id="show_routing", label="Show Agent Routing", initial=True),
        ]
    ).send()
    cl.user_session.set("settings", settings)

    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages with loading indicators and side-panel sources."""
    query = message.content.strip()
    if not query:
        return

    settings = cl.user_session.get("settings") or {}
    show_sources = settings.get("show_sources", True)
    show_routing = settings.get("show_routing", True)
    chat_history = cl.user_session.get("chat_history") or []

    start_time = time.time()

    # ── Send a loading message that updates live ──
    loading_msg = cl.Message(content="🤖 **Analyzing query...**\n\n⏳ Routing to specialized agents...")
    await loading_msg.send()

    # Track tool calls for live loading updates
    loading_lines = ["🤖 **Analyzing query...**", ""]
    loop = asyncio.get_event_loop()

    def on_tool_call(event_type, tool_name, args, chunk_count):
        """Callback from orchestrator thread — updates the loading message live."""
        if event_type == "start":
            display = TOOL_DISPLAY_NAMES.get(tool_name, f"🔍 Calling {tool_name}...")
            agent_label = TOOL_AGENT_LABELS.get(tool_name, "Agent")
            query_arg = args.get("query", args.get("topic", args.get("section_number", "")))
            loading_lines.append(f"{display}")
            loading_lines.append(f"   *{agent_label} — \"{query_arg[:60]}\"*")
            loading_msg.content = "\n".join(loading_lines)
            asyncio.run_coroutine_threadsafe(loading_msg.update(), loop)

        elif event_type == "done":
            agent_label = TOOL_AGENT_LABELS.get(tool_name, "Agent")
            loading_lines.append(f"   ✅ {agent_label} returned **{chunk_count}** chunks")
            loading_lines.append("")
            loading_msg.content = "\n".join(loading_lines)
            asyncio.run_coroutine_threadsafe(loading_msg.update(), loop)

    try:
        # Run orchestrator with callback for live updates
        answer, sources, tool_events = await run_orchestrator_async(
            query, chat_history, on_tool_call=on_tool_call
        )
        elapsed = time.time() - start_time

        # Gather agent info
        agents_used = list(dict.fromkeys(e["agent_label"] for e in tool_events))
        total_chunks = sum(e["chunk_count"] for e in tool_events)

        # ── Build final response ──
        response_parts = []

        # Agent routing header
        if show_routing and tool_events:
            agents_str = ", ".join(f"**{a}**" for a in agents_used)
            response_parts.append(
                f"*🤖 Agents: {agents_str} · "
                f"{total_chunks} chunks · {elapsed:.1f}s*\n\n---\n\n"
            )

        # Normalize citations: convert [SOURCE X] → [Source X] so Chainlit element matching works
        answer = re.sub(r'\[SOURCE\s+(\d+)', lambda m: f'[Source {m.group(1)}', answer, flags=re.IGNORECASE)

        response_parts.append(answer)

        # ── Build side-panel source elements ──
        # Match chatapp_full.py pattern exactly: name="Source X", display="side"
        # Chainlit makes elements clickable when their name appears in the message text
        clean_elements = []
        if show_sources and sources:
            for i, s in enumerate(sources[:20], 1):
                element_name = f"Source {i}"
                title = s.get("title", "Unknown Document")
                section = s.get("section", "N/A")
                year = s.get("year", "N/A")
                page = s.get("page", "N/A")
                text = s.get("text", "No content available.")
                doc_source = s.get("source", "N/A")

                # Only add if the LLM actually cited this source in the answer
                # Check case-insensitive: LLM may write [SOURCE 1] or [Source 1]
                answer_upper = answer.upper()
                if f"SOURCE {i}]" in answer_upper or f"SOURCE {i}," in answer_upper or f"SOURCE {i} " in answer_upper or f"SOURCE {i}." in answer_upper:
                    clean_elements.append(
                        cl.Text(
                            name=element_name,
                            content=f"**Source {i}: {title}**\n\n**File:** {doc_source}\n**Page:** {page}\n**Section:** {section}\n**Year:** {year}\n\n**Preview:**\n{text[:1500]}",
                            display="side"
                        )
                    )

        # ── Replace loading message with final response ──
        loading_msg.content = "".join(response_parts)
        loading_msg.elements = clean_elements
        await loading_msg.update()

        # ── Update chat history ──
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        loading_msg.content = f"**❌ Error:** {str(e)}"
        await loading_msg.update()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
