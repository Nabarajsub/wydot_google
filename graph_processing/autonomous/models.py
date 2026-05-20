"""Data models for the autonomous task system."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskStep:
    step_id: str
    tool: str
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    expected_output_type: str = "text"   # "draft" | "report" | "analysis" | "text"
    is_blocking: bool = False
    status: str = "PENDING"             # PENDING | RUNNING | DONE | ERROR | SKIPPED


@dataclass
class TaskPlan:
    task_id: str
    goal: str
    steps: List[TaskStep]
    created_at: float = 0.0


@dataclass
class StepResult:
    step_id: str
    status: str          # DONE | ERROR | AWAITING_FEEDBACK
    output: Any = None
    error: Optional[str] = None


@dataclass
class TaskWorkingMemory:
    task_id: str
    plan: TaskPlan
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    accumulated_context: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
