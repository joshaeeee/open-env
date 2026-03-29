"""Client for the OpenER environment."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ERAction, ERObservation, ERState


class OpenEREnv(EnvClient[ERAction, ERObservation, ERState]):
    """WebSocket client for the OpenER benchmark."""

    def _step_payload(self, action: ERAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ERObservation]:
        observation_payload = payload.get("observation", payload)
        observation = ERObservation(**observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ERState:
        return ERState(**payload)
