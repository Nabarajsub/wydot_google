"""Base class for all autonomous agents."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autonomous.models import TaskStep, TaskWorkingMemory, StepResult


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, step: "TaskStep", state: "TaskWorkingMemory") -> "StepResult":
        ...
