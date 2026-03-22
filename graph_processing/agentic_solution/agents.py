"""
Domain-specific agents for the WYDOT Knowledge Graph.
Each agent wraps scoped search tools and defines its Gemini function declarations.
"""
from typing import List, Dict, Optional
from . import tools


# ═══════════════════════════════════════════════════════════════
#  BASE AGENT
# ═══════════════════════════════════════════════════════════════

class BaseAgent:
    """Base class for all domain agents."""

    name: str = "base_agent"
    description: str = "Base agent"

    def search(self, query: str, year: Optional[int] = None,
               section: Optional[str] = None, limit: int = 15) -> List[Dict]:
        """Combined vector + fulltext scoped search."""
        vec = tools.scoped_vector_search(
            query, self.name, year=year, section=section, limit=limit
        )
        ft = tools.scoped_fulltext_search(
            query, self.name, year=year, limit=8
        )

        # Merge: vector first, then fulltext (deduplicated)
        seen = {r["id"] for r in vec}
        merged = list(vec)
        for r in ft:
            if r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)
        return merged[:limit]

    def get_section(self, section_number: str, year: Optional[int] = None) -> List[Dict]:
        return tools.get_section_content(section_number, self.name, year=year)

    def compare_versions(self, topic: str, year_old: int = 2010, year_new: int = 2021) -> Dict:
        return tools.compare_versions(topic, self.name, year_old, year_new)

    def format_results(self, chunks: List[Dict]) -> str:
        """Format search results into a context string for the LLM."""
        if not chunks:
            return "No results found."
        parts = []
        for i, c in enumerate(chunks, 1):
            src_info = f"[SOURCE {i}: {c.get('title', 'Unknown')}, {c.get('section', '')}, Year: {c.get('year', '?')}]"
            parts.append(f"{src_info}\n{c['text']}\n")
        return "\n---\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  DOMAIN AGENTS
# ═══════════════════════════════════════════════════════════════

class SpecsAgent(BaseAgent):
    name = "specs_agent"
    description = (
        "Search Wyoming Standard Specifications (2010, 2021) and Standard Plans. "
        "Covers: construction requirements, material specs, pay items, tolerances, "
        "thresholds, contractor obligations, penalties, warranties, insurance, "
        "change orders, disputes, certifications, environmental requirements, "
        "hazardous materials, temporary structures, work zone safety, surveying, "
        "paint, coatings, concrete, asphalt, aggregate, grinding, guardrails, "
        "culverts, erosion control, traffic control devices, welding, pile driving, "
        "any Section XXX reference."
    )


class ConstructionAgent(BaseAgent):
    name = "construction_agent"
    description = (
        "Search WYDOT Construction Manuals (2018-2026). "
        "Covers: field inspection procedures, project administration, "
        "how-to for engineers/inspectors, construction management processes, "
        "ADA ramp inspection, signal devices, earthwork procedures."
    )


class MaterialsAgent(BaseAgent):
    name = "materials_agent"
    description = (
        "Search Materials Testing Manuals (2019-2023). "
        "Covers: lab test procedures, sampling methods, testing frequencies, "
        "material acceptance, aggregate testing, density testing, asphalt content, "
        "concrete cylinder tests, profiler certification."
    )


class DesignAgent(BaseAgent):
    name = "design_agent"
    description = (
        "Search Road/Bridge Design Manuals and Design Guides. "
        "Covers: geometric design, horizontal/vertical alignment, cross sections, "
        "superelevation, sight distance, CADD drafting standards."
    )


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    description = (
        "Search Traffic Crash Reports (2020-2024), Highway Safety Plans, "
        "and Vulnerable Road User reports. "
        "Covers: crash statistics, fatalities by county/year/type, "
        "accident trends, impaired driving data, safety improvement programs."
    )


class BridgeAgent(BaseAgent):
    name = "bridge_agent"
    description = (
        "Search Bridge Program documents, Bridge Design Manuals, "
        "Bridge Plans, and Approach Slab details. "
        "Covers: bridge design, load ratings, load postings, bridge plans, "
        "structural details, deck replacement, approach slab design."
    )


class PlanningAgent(BaseAgent):
    name = "planning_agent"
    description = (
        "Search STIP (State Transportation Improvement Program), "
        "Corridor Studies, and Long Range Plans. "
        "Covers: project funding, planned projects, corridor studies, "
        "transportation improvement programming."
    )


class AdminAgent(BaseAgent):
    name = "admin_agent"
    description = (
        "Search WYDOT Annual Reports, Financial Reports, Operating Budgets. "
        "Covers: department accomplishments, organizational info, "
        "financial data, leadership, DBE goals, staffing."
    )


class GeneralAgent(BaseAgent):
    name = "general_agent"
    description = (
        "Search ALL documents when query doesn't fit a specific domain. "
        "Covers: driver licenses, vehicle registration, permits, forms, "
        "general WYDOT information, cross-domain queries."
    )

    def search(self, query: str, year: Optional[int] = None,
               section: Optional[str] = None, limit: int = 15) -> List[Dict]:
        """General agent uses global search (all documents)."""
        return tools.global_search(query, year=year, limit=limit)


# ── Agent registry ──
AGENT_REGISTRY = {
    "specs_agent": SpecsAgent(),
    "construction_agent": ConstructionAgent(),
    "materials_agent": MaterialsAgent(),
    "design_agent": DesignAgent(),
    "safety_agent": SafetyAgent(),
    "bridge_agent": BridgeAgent(),
    "planning_agent": PlanningAgent(),
    "admin_agent": AdminAgent(),
    "general_agent": GeneralAgent(),
}


def get_agent(name: str) -> BaseAgent:
    return AGENT_REGISTRY.get(name, AGENT_REGISTRY["general_agent"])
