from .base_agent import BaseAgent
from .drafting_agent import DraftingAgent
from .ok_agent import OkAgent
from .search_agent import SearchAgent

AGENT_REGISTRY = {
    "drafting_agent": DraftingAgent,
    "ok_agent": OkAgent,
    "search_agent": SearchAgent,
}

__all__ = ["BaseAgent", "DraftingAgent", "OkAgent", "SearchAgent", "AGENT_REGISTRY"]
