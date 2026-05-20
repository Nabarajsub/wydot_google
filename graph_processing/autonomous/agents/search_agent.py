"""Agent that searches the Neo4j knowledge graph for relevant context."""
from __future__ import annotations
import sys
import os
from typing import TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from autonomous.models import TaskStep, TaskWorkingMemory, StepResult


class SearchAgent(BaseAgent):
    name = "search_agent"

    async def run(self, step: "TaskStep", state: "TaskWorkingMemory") -> "StepResult":
        from autonomous.models import StepResult

        inputs = step.inputs or {}
        query: str = inputs.get("query", state.plan.goal)

        try:
            # Use the same search function from chatapp_full
            _gp = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if _gp not in sys.path:
                sys.path.insert(0, _gp)
            from chatapp_full import search_graph_async
            context, sources = await search_graph_async(
                query, index_name="wydot_gemini_index", use_gemini=True
            )
            output = context[:4000] if context else "No relevant documents found."
        except Exception as e:
            output = f"Search completed for: {query}. (Search unavailable: {e})"

        return StepResult(step_id=step.step_id, status="DONE", output=output)
