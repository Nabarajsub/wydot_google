"""DAG execution engine: runs TaskPlan steps with dependency resolution."""
from __future__ import annotations
import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING

from .models import StepResult, TaskStep, TaskWorkingMemory
from .agents import AGENT_REGISTRY

if TYPE_CHECKING:
    from .models import TaskPlan


async def execute_plan(
    plan: "TaskPlan",
    state: TaskWorkingMemory,
    on_step_start=None,
    on_step_done=None,
    on_hitl_gate=None,
) -> TaskWorkingMemory:
    """Execute a TaskPlan DAG, honouring dependencies and HITL gates."""
    completed: set = set()
    remaining = list(plan.steps)

    while remaining:
        # Find all steps whose dependencies are satisfied
        ready = [
            s for s in remaining
            if all(dep in completed for dep in s.depends_on)
        ]
        if not ready:
            break  # cyclic dependency or all blocked

        # Execute ready steps in parallel
        results = await asyncio.gather(
            *[_execute_step(s, state, on_step_start) for s in ready],
            return_exceptions=True,
        )

        for step, result in zip(ready, results):
            remaining.remove(step)

            if isinstance(result, Exception):
                sr = StepResult(step_id=step.step_id, status="ERROR",
                                error=str(result))
            elif isinstance(result, dict):
                sr = StepResult(
                    step_id=step.step_id,
                    status=result.get("status", "DONE"),
                    output=result.get("output"),
                    error=result.get("error"),
                )
            else:
                sr = result  # already a StepResult

            step.status = sr.status
            state.step_results[step.step_id] = sr

            if sr.output:
                state.accumulated_context.append(
                    f"[{step.description}]\n{str(sr.output)[:2000]}"
                )

            if on_step_done:
                await on_step_done(step, sr)

            # HITL gate: pause for human review on draft steps
            if sr.status == "DONE" and step.expected_output_type == "draft":
                if on_hitl_gate:
                    await on_hitl_gate(step, sr, state)
                return state  # pause — handler resumes via action callback

            completed.add(step.step_id)

    return state


async def _execute_step(
    step: TaskStep,
    state: TaskWorkingMemory,
    on_step_start=None,
) -> StepResult:
    if on_step_start:
        await on_step_start(step)

    agent_cls = AGENT_REGISTRY.get(step.tool)
    if not agent_cls:
        # Unknown tool — use ok_agent as fallback
        from .agents.ok_agent import OkAgent
        agent = OkAgent()
    else:
        agent = agent_cls()

    return await agent.run(step, state)
