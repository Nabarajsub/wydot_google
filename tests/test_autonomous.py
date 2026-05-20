"""Tests for the autonomous task system."""
import asyncio
import sys
import os
import pytest

# Ensure graph_processing is on path
_gp = os.path.join(os.path.dirname(__file__), "..", "graph_processing")
if _gp not in sys.path:
    sys.path.insert(0, _gp)


# ── Model tests ───────────────────────────────────────────────────────────────

def test_task_step_defaults():
    from autonomous.models import TaskStep
    step = TaskStep(step_id="s1", tool="ok_agent", description="test")
    assert step.status == "PENDING"
    assert step.expected_output_type == "text"
    assert step.depends_on == []


def test_task_plan_creation():
    from autonomous.models import TaskPlan, TaskStep
    steps = [TaskStep("s1", "ok_agent", "do thing")]
    plan = TaskPlan(task_id="t1", goal="test goal", steps=steps)
    assert plan.task_id == "t1"
    assert len(plan.steps) == 1


def test_working_memory_init():
    from autonomous.models import TaskWorkingMemory, TaskPlan, TaskStep
    plan = TaskPlan("t1", "goal", [TaskStep("s1", "ok_agent", "desc")])
    mem = TaskWorkingMemory(task_id="t1", plan=plan)
    assert mem.accumulated_context == []
    assert mem.corrections == []
    assert mem.step_results == {}


def test_step_result():
    from autonomous.models import StepResult
    r = StepResult(step_id="s1", status="DONE", output="hello")
    assert r.status == "DONE"
    assert r.output == "hello"
    assert r.error is None


# ── Episodic store tests ──────────────────────────────────────────────────────

def test_episodic_store_init(tmp_path):
    from autonomous.memory.episodic_store import EpisodicStore
    store = EpisodicStore(db_path=str(tmp_path / "ep.db"))
    assert (tmp_path / "ep.db").exists()


def test_episodic_store_save_and_retrieve(tmp_path):
    from autonomous.memory.episodic_store import EpisodicStore
    store = EpisodicStore(db_path=str(tmp_path / "ep.db"))
    store.save_episode("t1", "bridge inspection goal", "success", ["correction1"])
    results = store.get_similar("bridge inspection")
    assert len(results) >= 1
    assert results[0]["task_id"] == "t1"


def test_episodic_store_no_match(tmp_path):
    from autonomous.memory.episodic_store import EpisodicStore
    store = EpisodicStore(db_path=str(tmp_path / "ep.db"))
    store.save_episode("t1", "bridge inspection", "done")
    results = store.get_similar("completely unrelated xyz query")
    assert results == []


# ── Artifact manager tests ────────────────────────────────────────────────────

def test_save_text(tmp_path, monkeypatch):
    import autonomous.artifact_manager as am
    monkeypatch.setattr(am, "_ARTIFACTS_DIR", tmp_path)
    meta = am.save_text("task_abc", "Safety Memo", "# Hello\nContent here")
    assert meta.path.exists()
    assert meta.path.read_text() == "# Hello\nContent here"
    assert meta.title == "Safety Memo"


def test_save_text_slug(tmp_path, monkeypatch):
    import autonomous.artifact_manager as am
    monkeypatch.setattr(am, "_ARTIFACTS_DIR", tmp_path)
    meta = am.save_text("t1", "My Special! Report@2024", "content")
    assert " " not in meta.path.name
    assert "@" not in meta.path.name


# ── Planner tests ─────────────────────────────────────────────────────────────

def test_fallback_plan_draft_goal():
    from autonomous.planner import _fallback_plan
    plan = _fallback_plan("Draft a safety memo for I-80")
    tools = [s["tool"] for s in plan["steps"]]
    assert "drafting_agent" in tools
    draft_step = next(s for s in plan["steps"] if s["tool"] == "drafting_agent")
    assert draft_step["expected_output_type"] == "draft"


def test_fallback_plan_analysis_goal():
    from autonomous.planner import _fallback_plan
    plan = _fallback_plan("List at-risk projects for next 30 days")
    tools = [s["tool"] for s in plan["steps"]]
    assert "search_agent" in tools
    assert "drafting_agent" not in tools


def test_fallback_plan_step_ids():
    from autonomous.planner import _fallback_plan
    plan = _fallback_plan("Analyze bridge inspection data")
    ids = [s["step_id"] for s in plan["steps"]]
    assert ids[0] == "step_1"
    assert ids[1] == "step_2"


# ── Execution engine tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_simple_plan():
    from autonomous.models import TaskPlan, TaskStep, TaskWorkingMemory
    from autonomous.execution_engine import execute_plan

    steps_called = []

    class MockAgent:
        async def run(self, step, state):
            from autonomous.models import StepResult
            steps_called.append(step.step_id)
            return StepResult(step_id=step.step_id, status="DONE", output=f"result_{step.step_id}")

    import autonomous.execution_engine as ee
    original = ee.AGENT_REGISTRY.copy()
    ee.AGENT_REGISTRY["ok_agent"] = MockAgent

    plan = TaskPlan("t1", "test", [
        TaskStep("s1", "ok_agent", "step one", depends_on=[]),
        TaskStep("s2", "ok_agent", "step two", depends_on=["s1"]),
    ])
    state = TaskWorkingMemory("t1", plan)
    result = await execute_plan(plan, state)

    ee.AGENT_REGISTRY.update(original)
    assert "s1" in result.step_results
    assert "s2" in result.step_results
    assert result.step_results["s1"].status == "DONE"


@pytest.mark.asyncio
async def test_hitl_gate_pauses_on_draft():
    from autonomous.models import TaskPlan, TaskStep, TaskWorkingMemory, StepResult
    from autonomous.execution_engine import execute_plan

    hitl_triggered = []

    class DraftMock:
        async def run(self, step, state):
            return StepResult(step_id=step.step_id, status="DONE", output="draft content")

    async def fake_hitl(step, result, state):
        hitl_triggered.append(step.step_id)

    import autonomous.execution_engine as ee
    ee.AGENT_REGISTRY["drafting_agent"] = DraftMock

    plan = TaskPlan("t1", "write memo", [
        TaskStep("s1", "drafting_agent", "draft", expected_output_type="draft"),
    ])
    state = TaskWorkingMemory("t1", plan)
    await execute_plan(plan, state, on_hitl_gate=fake_hitl)
    assert "s1" in hitl_triggered


# ── Agent tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ok_agent_returns_done(monkeypatch):
    from autonomous.agents.ok_agent import OkAgent
    from autonomous.models import TaskStep, TaskPlan, TaskWorkingMemory

    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    class FakeModel:
        def generate_content(self, *a, **kw):
            class R:
                text = "summary result"
            return R()

    import autonomous.agents.ok_agent as oa
    import unittest.mock as mock
    with mock.patch("google.generativeai.GenerativeModel", return_value=FakeModel()):
        with mock.patch("google.generativeai.configure"):
            agent = OkAgent()
            step = TaskStep("s1", "ok_agent", "do something", inputs={"description": "test"})
            plan = TaskPlan("t1", "goal", [step])
            state = TaskWorkingMemory("t1", plan)
            result = await agent.run(step, state)
    assert result.status == "DONE"
    assert result.output == "summary result"


@pytest.mark.asyncio
async def test_ok_agent_fallback_on_error(monkeypatch):
    from autonomous.agents.ok_agent import OkAgent
    from autonomous.models import TaskStep, TaskPlan, TaskWorkingMemory
    import unittest.mock as mock

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with mock.patch("google.generativeai.configure"):
        with mock.patch("google.generativeai.GenerativeModel", side_effect=Exception("no llm")):
            agent = OkAgent()
            step = TaskStep("s1", "ok_agent", "test", inputs={"description": "test"})
            plan = TaskPlan("t1", "goal", [step])
            state = TaskWorkingMemory("t1", plan)
            result = await agent.run(step, state)
    assert result.status == "DONE"
    assert "completed" in result.output.lower() or "no llm" in result.output.lower()


@pytest.mark.asyncio
async def test_drafting_agent_includes_revision_prompt(monkeypatch):
    from autonomous.agents.drafting_agent import DraftingAgent
    from autonomous.models import TaskStep, TaskPlan, TaskWorkingMemory
    import unittest.mock as mock

    captured = []

    class FakeModel:
        def generate_content(self, prompt, **kw):
            captured.append(prompt)
            class R:
                text = "revised draft"
            return R()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with mock.patch("google.generativeai.configure"):
        with mock.patch("google.generativeai.GenerativeModel", return_value=FakeModel()):
            agent = DraftingAgent()
            step = TaskStep(
                "s1", "drafting_agent", "draft memo",
                inputs={"topic": "safety", "revision_feedback": "make it shorter"},
                expected_output_type="draft",
            )
            plan = TaskPlan("t1", "write memo", [step])
            state = TaskWorkingMemory("t1", plan)
            result = await agent.run(step, state)

    assert result.status == "DONE"
    assert len(captured) == 1
    assert "REVISION INSTRUCTIONS" in captured[0]
    assert "make it shorter" in captured[0]


# ── Task handler unit tests ───────────────────────────────────────────────────

def test_task_id_from_payload():
    import unittest.mock as mock
    with mock.patch("chainlit.Action") as MockAction:
        action = mock.MagicMock()
        action.payload = {"task_id": "task_abc123"}
        from autonomous.task_handler import _task_id_from
        assert _task_id_from(action) == "task_abc123"


def test_task_id_from_none_payload():
    import unittest.mock as mock
    action = mock.MagicMock()
    action.payload = None
    action.value = "fallback_id"
    from autonomous.task_handler import _task_id_from
    result = _task_id_from(action)
    assert result == "fallback_id"


# ── Short-ack guard logic test ────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_ack", [
    ("ok thanks", True),
    ("looks good to me", True),
    ("approve", True),
    ("Draft a safety memo for the I-80 corridor", False),
    ("ok lets approve and continue", True),
    ("Generate a weekly status report for district 4", False),
    ("got it continue", True),
])
def test_short_ack_guard(text, expected_ack):
    _ack_words = {"ok", "thanks", "approve", "continue", "looks", "good",
                  "great", "done", "fine", "noted", "got", "it", "sure"}
    words = text.lower().split()
    is_ack = len(words) <= 6 and bool(_ack_words & set(words))
    assert is_ack == expected_ack, f"Failed for: {text!r}"
