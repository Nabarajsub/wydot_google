"""LLM-based task planner: converts a user goal into a TaskPlan DAG."""
from __future__ import annotations
import json
import os
import time
import uuid
from typing import List

from .models import TaskPlan, TaskStep


_PLANNER_PROMPT = """You are a task planner for WYDOT (Wyoming DOT) engineering workflows.

Convert the user's goal into a structured execution plan with 2-5 steps.

Available tools:
- search_agent: Search the WYDOT knowledge graph for relevant documents and context
- ok_agent: Perform analysis, summarization, or data gathering
- drafting_agent: Write a formal document (memo, report, change order, inspection summary)

Return ONLY valid JSON matching this schema (no markdown, no explanation):
{{
  "goal": "<restate goal concisely>",
  "steps": [
    {{
      "step_id": "step_1",
      "tool": "<tool_name>",
      "description": "<what this step does>",
      "inputs": {{}},
      "depends_on": [],
      "expected_output_type": "text",
      "is_blocking": false
    }}
  ]
}}

Rules:
- The LAST step that produces the final deliverable should use drafting_agent if the goal is to write a document
- Set expected_output_type="draft" on the drafting step
- Steps that can run in parallel share no depends_on entries
- Keep step_ids sequential: step_1, step_2, ...

User goal: {goal}
"""


async def create_plan(goal: str) -> TaskPlan:
    task_id = f"task_{uuid.uuid4().hex[:8]}"

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(
            _PLANNER_PROMPT.format(goal=goal),
            generation_config={"temperature": 0.2},
        )
        raw = resp.text.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception as e:
        print(f"[Planner] LLM error, using fallback: {e}")
        data = _fallback_plan(goal)

    steps: List[TaskStep] = []
    for s in data.get("steps", []):
        steps.append(TaskStep(
            step_id=s.get("step_id", f"step_{len(steps)+1}"),
            tool=s.get("tool", "ok_agent"),
            description=s.get("description", ""),
            inputs=s.get("inputs", {}),
            depends_on=s.get("depends_on", []),
            expected_output_type=s.get("expected_output_type", "text"),
            is_blocking=s.get("is_blocking", False),
        ))

    return TaskPlan(
        task_id=task_id,
        goal=data.get("goal", goal),
        steps=steps,
        created_at=time.time(),
    )


def _fallback_plan(goal: str) -> dict:
    is_draft = any(w in goal.lower() for w in
                   ["draft", "write", "memo", "report", "letter", "summary"])
    steps = [
        {
            "step_id": "step_1",
            "tool": "search_agent",
            "description": f"Search WYDOT knowledge base for context on: {goal}",
            "inputs": {"query": goal},
            "depends_on": [],
            "expected_output_type": "text",
            "is_blocking": False,
        }
    ]
    if is_draft:
        steps.append({
            "step_id": "step_2",
            "tool": "drafting_agent",
            "description": f"Draft the document: {goal}",
            "inputs": {"topic": goal, "doc_type": "memo"},
            "depends_on": ["step_1"],
            "expected_output_type": "draft",
            "is_blocking": False,
        })
    else:
        steps.append({
            "step_id": "step_2",
            "tool": "ok_agent",
            "description": f"Analyze and summarize findings for: {goal}",
            "inputs": {"description": goal},
            "depends_on": ["step_1"],
            "expected_output_type": "text",
            "is_blocking": False,
        })
    return {"goal": goal, "steps": steps}
