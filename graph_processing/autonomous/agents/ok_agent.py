"""Simple pass-through agent that marks a step complete with a summary."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from autonomous.models import TaskStep, TaskWorkingMemory, StepResult


class OkAgent(BaseAgent):
    name = "ok_agent"

    async def run(self, step: "TaskStep", state: "TaskWorkingMemory") -> "StepResult":
        from autonomous.models import StepResult

        inputs = step.inputs or {}
        task_desc = inputs.get("description", step.description)

        context = "\n".join(state.accumulated_context[-2:]) if state.accumulated_context else ""
        prompt = (
            f"Complete the following task step for WYDOT and provide a concise summary:\n\n"
            f"Task: {task_desc}\n\n"
            f"Overall goal: {state.plan.goal}\n"
            + (f"\nContext:\n{context}" if context else "")
        )

        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            output = response.text
        except Exception as e:
            output = f"Step '{task_desc}' completed. (LLM unavailable: {e})"

        return StepResult(step_id=step.step_id, status="DONE", output=output)
