"""Chainlit-facing handlers for the autonomous task system."""
from __future__ import annotations
import asyncio
import time
from typing import Optional

import chainlit as cl

from .models import TaskWorkingMemory, StepResult, TaskStep
from .planner import create_plan
from .execution_engine import execute_plan
from .artifact_manager import save_text


# ── helpers ──────────────────────────────────────────────────────────────────

def _task_id_from(action: cl.Action) -> Optional[str]:
    """Extract task_id from action payload (Chainlit 2.9.6+)."""
    if action.payload and isinstance(action.payload, dict):
        return action.payload.get("task_id")
    # fallback for older versions
    return getattr(action, "value", None)


async def _show_draft_for_review(draft: str, task_id: str):
    """Send draft in ≤6000-char chunks then Approve/Revise buttons."""
    chunk_size = 6000
    for i in range(0, len(draft), chunk_size):
        await cl.Message(content=draft[i:i + chunk_size]).send()

    await cl.Message(
        content="---\n**Review the draft above. Approve to save, or request revisions.**",
        actions=[
            cl.Action(name="approve_draft",
                      payload={"task_id": task_id},
                      label="✅ Approve & Save"),
            cl.Action(name="revise_draft",
                      payload={"task_id": task_id},
                      label="✏️ Request Revision"),
        ],
    ).send()


async def _finish_task(state: TaskWorkingMemory):
    """Display final non-draft output and persist episode."""
    last_output = None
    for step in reversed(state.plan.steps):
        sr = state.step_results.get(step.step_id)
        if sr and sr.status == "DONE" and sr.output:
            last_output = sr.output
            break

    if last_output:
        await cl.Message(content=f"**Result:**\n\n{last_output}").send()

    await cl.Message(
        content=f"✅ **Task complete:** {state.plan.goal}"
    ).send()

    try:
        from .memory.episodic_store import EpisodicStore
        store = EpisodicStore()
        store.save_episode(
            task_id=state.task_id,
            goal=state.plan.goal,
            outcome=str(last_output)[:500] if last_output else "completed",
            corrections=state.corrections,
        )
    except Exception as e:
        print(f"[task_handler] episodic store error: {e}")


# ── main entry point ──────────────────────────────────────────────────────────

async def start_task(goal: str):
    """Called from on_message when autonomous mode is on and a new task is detected."""
    t0 = time.perf_counter()

    plan = await create_plan(goal)
    state = TaskWorkingMemory(task_id=plan.task_id, plan=plan)
    cl.user_session.set("current_task", state)

    # Display the plan
    step_lines = "\n".join(
        f"  {i+1}. **{s.tool}** — {s.description}"
        for i, s in enumerate(plan.steps)
    )
    await cl.Message(
        content=(
            f"### 📋 Autonomous Task Plan\n\n"
            f"**Goal:** {plan.goal}\n\n"
            f"**Steps:**\n{step_lines}\n\n"
            f"*Estimated: {len(plan.steps) * 10}–{len(plan.steps) * 30}s*"
        ),
        actions=[
            cl.Action(name="proceed_task",
                      payload={"task_id": plan.task_id},
                      label="▶ Proceed"),
            cl.Action(name="cancel_task",
                      payload={"task_id": plan.task_id},
                      label="✕ Cancel"),
        ],
    ).send()


# ── action callbacks (registered in chatapp_full.py) ─────────────────────────

async def on_proceed_task(action: cl.Action):
    task_id = _task_id_from(action)
    state: Optional[TaskWorkingMemory] = cl.user_session.get("current_task")
    if not state or state.task_id != task_id:
        await cl.Message(content="⚠️ Task not found. Please start a new task.").send()
        return

    step_timing: dict = {}

    async def on_step_start(step: TaskStep):
        step_timing[step.step_id] = time.perf_counter()
        await cl.Message(content=f"⚙️ **{step.description}**…").send()

    async def on_step_done(step: TaskStep, result: StepResult):
        elapsed = time.perf_counter() - step_timing.get(step.step_id, time.perf_counter())
        status = "✅" if result.status == "DONE" else "❌"
        await cl.Message(
            content=f"{status} {step.description} done ({elapsed*1000:.0f}ms)"
        ).send()

    async def on_hitl_gate(step: TaskStep, result: StepResult, st: TaskWorkingMemory):
        cl.user_session.set("current_task", st)
        draft = str(result.output or "")
        await _show_draft_for_review(draft, st.task_id)

    try:
        state = await execute_plan(
            state.plan, state,
            on_step_start=on_step_start,
            on_step_done=on_step_done,
            on_hitl_gate=on_hitl_gate,
        )
        cl.user_session.set("current_task", state)

        # If no HITL gate triggered, task ran to completion
        all_done = all(
            state.step_results.get(s.step_id, StepResult("", "PENDING")).status == "DONE"
            for s in state.plan.steps
        )
        if all_done:
            await _finish_task(state)
    except Exception as e:
        await cl.Message(content=f"❌ Task failed: {e}").send()
        print(f"[task_handler] execute error: {e}")


async def on_cancel_task(action: cl.Action):
    cl.user_session.set("current_task", None)
    await cl.Message(content="🚫 Task cancelled.").send()


async def on_approve_draft(action: cl.Action):
    task_id = _task_id_from(action)
    state: Optional[TaskWorkingMemory] = cl.user_session.get("current_task")
    if not state:
        await cl.Message(content="⚠️ No active task.").send()
        return

    # Find the draft content
    draft_content = ""
    draft_step_id = None
    for step in state.plan.steps:
        if step.expected_output_type == "draft":
            sr = state.step_results.get(step.step_id)
            if sr and sr.output:
                draft_content = str(sr.output)
                draft_step_id = step.step_id
                break

    if not draft_content:
        await cl.Message(content="⚠️ Draft content not found.").send()
        return

    # Save artifact
    try:
        title = state.plan.goal[:50]
        meta = save_text(state.task_id, title, draft_content)
        await cl.Message(
            content=f"📎 Draft saved as **{meta.path.name}**",
            elements=[cl.File(name=meta.path.name, path=str(meta.path),
                              display="inline")],
        ).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Could not save file: {e}").send()

    # Mark remaining steps complete and finish
    for step in state.plan.steps:
        if step.step_id not in state.step_results:
            state.step_results[step.step_id] = StepResult(
                step_id=step.step_id, status="DONE", output="skipped after approval"
            )
    cl.user_session.set("current_task", None)
    await cl.Message(content=f"✅ **Task complete:** {state.plan.goal}").send()


async def on_revise_draft(action: cl.Action):
    task_id = _task_id_from(action)
    state: Optional[TaskWorkingMemory] = cl.user_session.get("current_task")
    if not state:
        await cl.Message(content="⚠️ No active task.").send()
        return

    cl.user_session.set("awaiting_revision_for", state.task_id)
    await cl.Message(
        content="✏️ **What changes would you like?** "
                "Describe the revision and I'll regenerate the draft."
    ).send()


# ── revision flow ─────────────────────────────────────────────────────────────

async def handle_revision_instructions(feedback: str):
    """Re-run the draft step with revision feedback injected."""
    state: Optional[TaskWorkingMemory] = cl.user_session.get("current_task")
    if not state:
        await cl.Message(content="⚠️ No active task to revise.").send()
        return

    state.corrections.append(feedback)

    # Find the draft step and reset it
    draft_step = None
    for step in state.plan.steps:
        if step.expected_output_type == "draft":
            draft_step = step
            break

    if not draft_step:
        await cl.Message(content="⚠️ No draft step found.").send()
        return

    # Remove old draft result and inject feedback
    old_result = state.step_results.pop(draft_step.step_id, None)
    if old_result and old_result.output:
        # Remove from accumulated context
        old_snippet = str(old_result.output)[:200]
        state.accumulated_context = [
            c for c in state.accumulated_context
            if old_snippet not in c
        ]

    draft_step.inputs["revision_feedback"] = feedback
    draft_step.status = "PENDING"

    cl.user_session.set("awaiting_revision_for", None)
    await cl.Message(content=f"🔄 **Revising draft** based on: _{feedback}_…").send()

    # Re-run just the draft step
    try:
        from .agents import AGENT_REGISTRY
        agent = AGENT_REGISTRY.get(draft_step.tool, AGENT_REGISTRY["drafting_agent"])()
        result = await agent.run(draft_step, state)
        draft_step.status = result.status
        state.step_results[draft_step.step_id] = result

        if result.status == "DONE" and result.output:
            state.accumulated_context.append(
                f"[Revised {draft_step.description}]\n{str(result.output)[:2000]}"
            )
            cl.user_session.set("current_task", state)
            await _show_draft_for_review(str(result.output), state.task_id)
        else:
            await cl.Message(content=f"❌ Revision failed: {result.error}").send()
    except Exception as e:
        await cl.Message(content=f"❌ Revision error: {e}").send()


async def handle_correction(correction: str):
    """Apply a mid-task correction to remaining steps."""
    state: Optional[TaskWorkingMemory] = cl.user_session.get("current_task")
    if not state:
        return False

    state.corrections.append(correction)
    # Inject correction context so upcoming steps see it
    state.accumulated_context.append(f"[User correction]: {correction}")
    cl.user_session.set("current_task", state)
    await cl.Message(content=f"✅ Correction noted: _{correction}_").send()
    return True
