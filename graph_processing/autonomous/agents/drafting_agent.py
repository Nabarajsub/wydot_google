"""Agent that produces a written draft (memo, report, change order, etc.)."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from autonomous.models import TaskStep, TaskWorkingMemory, StepResult


class DraftingAgent(BaseAgent):
    name = "drafting_agent"

    async def run(self, step: "TaskStep", state: "TaskWorkingMemory") -> "StepResult":
        from autonomous.models import StepResult

        inputs = step.inputs or {}
        topic: str = inputs.get("topic", state.plan.goal)
        doc_type: str = inputs.get("doc_type", "memo")
        context_hint: str = inputs.get("context_hint", "")
        revision_feedback: Optional[str] = inputs.get("revision_feedback")

        context_block = ""
        if state.accumulated_context:
            context_block = "\n\n## Background from prior steps\n" + \
                            "\n---\n".join(state.accumulated_context[-3:])

        revision_block = ""
        if revision_feedback:
            revision_block = (
                f"\n\n## REVISION INSTRUCTIONS — MUST FOLLOW\n"
                f"The user has reviewed the previous draft and requested changes:\n"
                f"> {revision_feedback}\n\n"
                f"You MUST incorporate these changes. Do NOT simply reproduce the prior "
                f"draft verbatim. The revised output must be meaningfully different."
            )

        prompt = (
            f"You are a professional technical writer for WYDOT "
            f"(Wyoming Department of Transportation).\n\n"
            f"Write a professional {doc_type} about: {topic}\n"
            f"{context_hint}{context_block}{revision_block}\n\n"
            f"Requirements:\n"
            f"- Use WYDOT formal tone\n"
            f"- Include relevant technical details\n"
            f"- Add section headings\n"
            f"- Be thorough but concise\n"
        )

        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            draft_text = response.text
        except Exception as e:
            draft_text = (
                f"# {doc_type.title()}: {topic}\n\n"
                f"*[Draft generation failed: {e}. "
                f"Please review and complete manually.]*\n\n"
                f"## Section 1\n\nContent here.\n\n"
                f"## Section 2\n\nContent here.\n"
            )

        return StepResult(
            step_id=step.step_id,
            status="DONE",
            output=draft_text,
        )
