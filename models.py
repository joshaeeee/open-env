"""Typed models for the OpenER benchmark."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


Difficulty = Literal["easy", "medium", "hard"]
PatientLocation = Literal["waiting", "bed", "ct", "lab", "discharged", "admitted"]
Disposition = Literal["home", "admit"]


class VitalsView(BaseModel):
    hr: int = Field(..., description="Heart rate in beats per minute")
    sbp: int = Field(..., description="Systolic blood pressure")
    dbp: int = Field(..., description="Diastolic blood pressure")
    rr: int = Field(..., description="Respiratory rate")
    o2_sat: int = Field(..., description="Peripheral oxygen saturation percentage")
    temp_c: float = Field(..., description="Temperature in Celsius")
    gcs: int = Field(..., description="Glasgow Coma Scale score")


class PatientView(BaseModel):
    patient_id: str
    chief_complaint: str
    age: int
    arrival_minute: int
    wait_time_min: int
    assigned_esi: int = Field(..., ge=1, le=5)
    location: PatientLocation
    vitals: VitalsView
    pain_scale: int = Field(..., ge=0, le=10)
    history: list[str] = Field(default_factory=list)
    completed_tests: dict[str, str | int | float] = Field(default_factory=dict)
    news2_score: int
    qsofa_score: int


class ResourceView(BaseModel):
    beds_available: int
    beds_total: int
    ct_available_in_min: int
    lab_queue_length: int
    specialist_available: dict[str, bool] = Field(default_factory=dict)


class PatientCommand(BaseModel):
    patient_id: str
    assign_bed: bool = False
    new_esi: int | None = Field(default=None, ge=1, le=5)
    order_tests: list[str] = Field(default_factory=list)
    disposition: Disposition | None = None
    call_specialist: str | None = None


class ERAction(Action):
    commands: list[PatientCommand] = Field(default_factory=list)


class ERObservation(Observation):
    task_id: str
    difficulty: Difficulty
    shift_minute: int
    patients: list[PatientView] = Field(default_factory=list)
    resources: ResourceView
    alerts: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)


class ERState(State):
    task_id: str = "easy_single_critical"
    difficulty: Difficulty = "easy"
    seed: int = 0
    shift_minute: int = 0
    total_seen: int = 0
    total_discharged: int = 0
    total_admitted: int = 0
    adverse_events: int = 0
    lwbs_count: int = 0
    total_cost: float = 0.0
    benchmark_score: float | None = None
    benchmark_breakdown: dict[str, float] = Field(default_factory=dict)
