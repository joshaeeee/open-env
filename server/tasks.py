"""Official OpenER task configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    start_hour: float
    max_steps: int = 96
    beds_total: int
    lab_slots: int
    ct_slots: int
    initial_patient_count: int
    arrival_profile: str
    required_hidden_cases: list[str] = Field(default_factory=list)
    grader_weights: dict[str, float] = Field(default_factory=dict)


OFFICIAL_TASKS: dict[str, TaskConfig] = {
    "easy_single_critical": TaskConfig(
        task_id="easy_single_critical",
        difficulty="easy",
        start_hour=9.0,
        beds_total=6,
        lab_slots=2,
        ct_slots=1,
        initial_patient_count=7,
        arrival_profile="low",
        required_hidden_cases=["stemi"],
        grader_weights={
            "patient_safety": 0.45,
            "timeliness": 0.25,
            "diagnostic_quality": 0.20,
            "resource_stewardship": 0.10,
        },
    ),
    "medium_evening_rush": TaskConfig(
        task_id="medium_evening_rush",
        difficulty="medium",
        start_hour=17.0,
        beds_total=7,
        lab_slots=2,
        ct_slots=1,
        initial_patient_count=11,
        arrival_profile="medium",
        required_hidden_cases=["stemi", "appendicitis"],
        grader_weights={
            "patient_safety": 0.40,
            "timeliness": 0.25,
            "diagnostic_quality": 0.20,
            "resource_stewardship": 0.15,
        },
    ),
    "hard_capacity_crunch": TaskConfig(
        task_id="hard_capacity_crunch",
        difficulty="hard",
        start_hour=19.0,
        beds_total=5,
        lab_slots=2,
        ct_slots=1,
        initial_patient_count=17,
        arrival_profile="high",
        required_hidden_cases=["stemi", "sepsis", "head_bleed", "appendicitis"],
        grader_weights={
            "patient_safety": 0.40,
            "timeliness": 0.25,
            "diagnostic_quality": 0.20,
            "resource_stewardship": 0.15,
        },
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in OFFICIAL_TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return OFFICIAL_TASKS[task_id].model_copy(deep=True)
