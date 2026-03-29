"""Core environment logic for OpenER."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ERAction,
        ERObservation,
        ERState,
        PatientView,
        ResourceView,
        VitalsView,
    )
    from .data import (
        CRITICAL_NEWS2_THRESHOLD,
        ESI_WAIT_TARGET_MIN,
        HIGH_NEWS2_THRESHOLD,
        HIGH_QSOFA_THRESHOLD,
        SPECIALISTS,
        TEST_CATALOG,
        arrival_rate,
        compute_news2,
        compute_qsofa,
        generate_test_result,
        sample_patient_blueprint,
    )
    from .rubrics import grade_episode
    from .tasks import TaskConfig, get_task_config
except ImportError:  # pragma: no cover - direct source execution fallback
    from models import (  # type: ignore
        ERAction,
        ERObservation,
        ERState,
        PatientView,
        ResourceView,
        VitalsView,
    )
    from server.data import (  # type: ignore
        CRITICAL_NEWS2_THRESHOLD,
        ESI_WAIT_TARGET_MIN,
        HIGH_NEWS2_THRESHOLD,
        HIGH_QSOFA_THRESHOLD,
        SPECIALISTS,
        TEST_CATALOG,
        arrival_rate,
        compute_news2,
        compute_qsofa,
        generate_test_result,
        sample_patient_blueprint,
    )
    from server.rubrics import grade_episode  # type: ignore
    from server.tasks import TaskConfig, get_task_config  # type: ignore


@dataclass
class PendingTest:
    patient_id: str
    test_name: str
    complete_minute: int


@dataclass
class InternalPatient:
    patient_id: str
    chief_complaint: str
    age: int
    pain_scale: int
    history: list[str]
    arrival_minute: int
    vitals: dict[str, int | float]
    diagnosis: str
    true_esi: int
    trajectory: str
    required_tests: set[str]
    correct_disposition: str
    recommended_specialist: str | None
    assigned_esi: int
    location: str = "waiting"
    completed_tests: dict[str, str | int | float] = field(default_factory=dict)
    ordered_tests: set[str] = field(default_factory=set)
    unnecessary_tests: set[str] = field(default_factory=set)
    consults_called: set[str] = field(default_factory=set)
    bed_assigned_minute: int | None = None
    urgent_wait_breached: bool = False
    terminal_outcome: str | None = None
    unsafe_disposition: bool = False
    correct_disposition_hit: bool = False

    @property
    def is_active(self) -> bool:
        return self.terminal_outcome is None


class ERTriageEnvironment(Environment[ERAction, ERObservation, ERState]):
    """Single-agent emergency department triage benchmark."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        self._task: TaskConfig = get_task_config("easy_single_critical")
        self._patients: dict[str, InternalPatient] = {}
        self._future_arrivals: list[tuple[int, str | None]] = []
        self._pending_tests: list[PendingTest] = []
        self._state = ERState()
        self._events: list[str] = []
        self._alerts: list[str] = []
        self._step_reward: float = 0.0
        self._reward_components: dict[str, float] = {}
        self._ct_busy_until = 0
        self._lab_busy_until: list[int] = []
        self._specialist_busy_until: dict[str, int] = {}

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> ERObservation:
        task_id = kwargs.get("task_id", "easy_single_critical")
        self._task = get_task_config(task_id)
        self._rng = random.Random(seed if seed is not None else 0)
        self._patients = {}
        self._future_arrivals = []
        self._pending_tests = []
        self._events = []
        self._alerts = []
        self._reset_reward_components()
        self._ct_busy_until = 0
        self._lab_busy_until = [0 for _ in range(self._task.lab_slots)]
        self._specialist_busy_until = {name: 0 for name in SPECIALISTS}
        self._state = ERState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            seed=seed or 0,
            shift_minute=0,
        )

        forced = list(self._task.required_hidden_cases)
        for index in range(self._task.initial_patient_count):
            forced_diagnosis = forced[index] if index < len(forced) else None
            patient = self._create_patient(arrival_minute=0, forced_diagnosis=forced_diagnosis)
            self._patients[patient.patient_id] = patient

        self._future_arrivals = self._generate_future_arrivals()
        self._recount_state()
        self._events.append(f"Shift started for task '{self._task.task_id}'.")
        return self._build_observation(done=False)

    def step(
        self,
        action: ERAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ERObservation:
        del timeout_s, kwargs
        self._events = []
        self._alerts = []
        self._reset_reward_components()
        self._state.step_count += 1

        self._process_commands(action)
        self._advance_time()
        self._complete_tests()
        self._release_specialists()
        self._spawn_arrivals()
        self._update_deterioration()
        self._check_wait_breaches_and_lwbs()
        self._recount_state()

        done = self._state.step_count >= self._task.max_steps
        if done:
            result = grade_episode(self._task, self._patients.values())
            self._state.benchmark_score = result.score
            self._state.benchmark_breakdown = result.breakdown
        return self._build_observation(done=done)

    @property
    def state(self) -> ERState:
        return self._state

    def _create_patient(
        self,
        arrival_minute: int,
        forced_diagnosis: str | None = None,
    ) -> InternalPatient:
        blueprint = sample_patient_blueprint(self._rng, forced_diagnosis=forced_diagnosis)
        patient_id = f"pt_{len(self._patients) + 1:03d}_{arrival_minute}"
        return InternalPatient(
            patient_id=patient_id,
            chief_complaint=blueprint["chief_complaint"],
            age=blueprint["age"],
            pain_scale=blueprint["pain_scale"],
            history=blueprint["history"],
            arrival_minute=arrival_minute,
            vitals=blueprint["vitals"],
            diagnosis=blueprint["diagnosis"],
            true_esi=blueprint["true_esi"],
            trajectory=blueprint["trajectory"],
            required_tests=set(blueprint["required_tests"]),
            correct_disposition=blueprint["correct_disposition"],
            recommended_specialist=blueprint["specialist"],
            assigned_esi=min(5, blueprint["true_esi"] + 1),
        )

    def _generate_future_arrivals(self) -> list[tuple[int, str | None]]:
        future: list[tuple[int, str | None]] = []
        for step in range(1, self._task.max_steps):
            minute = step * 5
            arrivals = arrival_rate(self._task.arrival_profile, step)
            for _ in range(arrivals):
                future.append((minute, None))
        return future

    def _reset_reward_components(self) -> None:
        self._reward_components = {
            "patient_safety_delta": 0.0,
            "timeliness_delta": 0.0,
            "diagnostic_quality_delta": 0.0,
            "resource_efficiency_delta": 0.0,
            "disposition_quality_delta": 0.0,
        }
        self._step_reward = 0.0

    def _bump_reward(self, component: str, delta: float) -> None:
        self._reward_components[component] += delta

    def _finalize_step_reward(self) -> float:
        weighted = (
            0.35 * self._reward_components["patient_safety_delta"]
            + 0.25 * self._reward_components["timeliness_delta"]
            + 0.20 * self._reward_components["diagnostic_quality_delta"]
            + 0.10 * self._reward_components["resource_efficiency_delta"]
            + 0.10 * self._reward_components["disposition_quality_delta"]
        )
        self._step_reward = round(weighted, 4)
        return self._step_reward

    def _process_commands(self, action: ERAction) -> None:
        seen: set[str] = set()
        for command in action.commands:
            if command.patient_id in seen:
                self._events.append(f"Ignored duplicate command for {command.patient_id}.")
                continue
            seen.add(command.patient_id)
            patient = self._patients.get(command.patient_id)
            if patient is None or not patient.is_active:
                self._events.append(f"Ignored command for unknown or inactive patient {command.patient_id}.")
                continue

            if command.new_esi is not None:
                old_undertriage = max(0, patient.assigned_esi - patient.true_esi)
                new_undertriage = max(0, command.new_esi - patient.true_esi)
                old_overtriage = max(0, patient.true_esi - patient.assigned_esi)
                new_overtriage = max(0, patient.true_esi - command.new_esi)
                patient.assigned_esi = command.new_esi
                if new_undertriage < old_undertriage:
                    self._bump_reward(
                        "patient_safety_delta",
                        1.2 * float(old_undertriage - new_undertriage),
                    )
                elif new_undertriage > old_undertriage:
                    self._bump_reward(
                        "patient_safety_delta",
                        -1.6 * float(new_undertriage - old_undertriage),
                    )

                if new_overtriage < old_overtriage:
                    self._bump_reward(
                        "resource_efficiency_delta",
                        0.3 * float(old_overtriage - new_overtriage),
                    )
                elif new_overtriage > old_overtriage:
                    self._bump_reward(
                        "resource_efficiency_delta",
                        -0.2 * float(new_overtriage - old_overtriage),
                    )
                self._events.append(f"Updated ESI for {patient.patient_id} to {command.new_esi}.")

            if command.assign_bed:
                self._assign_bed(patient)

            for test_name in command.order_tests:
                self._order_test(patient, test_name)

            if command.call_specialist:
                self._call_specialist(patient, command.call_specialist)

            if command.disposition:
                self._apply_disposition(patient, command.disposition)

    def _assign_bed(self, patient: InternalPatient) -> None:
        if patient.location != "waiting":
            self._events.append(f"{patient.patient_id} is not waiting; bed assignment skipped.")
            return
        if self._beds_in_use() >= self._task.beds_total:
            self._events.append(f"No beds available for {patient.patient_id}.")
            return
        patient.location = "bed"
        patient.bed_assigned_minute = self._state.shift_minute
        news2 = compute_news2(patient.vitals)
        qsofa = compute_qsofa(patient.vitals)
        if patient.true_esi <= 2 or news2 >= HIGH_NEWS2_THRESHOLD or qsofa >= HIGH_QSOFA_THRESHOLD:
            self._bump_reward("timeliness_delta", 2.5)
            self._bump_reward("patient_safety_delta", 0.8)
        else:
            self._bump_reward("timeliness_delta", 0.3)
            self._bump_reward("resource_efficiency_delta", -0.1)
        self._events.append(f"Assigned bed to {patient.patient_id}.")

    def _order_test(self, patient: InternalPatient, test_name: str) -> None:
        if test_name not in TEST_CATALOG:
            self._events.append(f"Unknown test '{test_name}' for {patient.patient_id}.")
            return
        if test_name in patient.ordered_tests:
            self._step_reward -= 0.1
            self._events.append(f"Duplicate test '{test_name}' ignored for {patient.patient_id}.")
            return

        patient.ordered_tests.add(test_name)
        spec = TEST_CATALOG[test_name]
        complete_minute = self._state.shift_minute + int(spec["turnaround_min"])
        resource = spec["resource"]
        if resource == "ct":
            complete_minute = max(complete_minute, self._ct_busy_until + int(spec["turnaround_min"]))
            self._ct_busy_until = complete_minute
        elif resource == "lab" and self._lab_busy_until:
            next_slot = min(range(len(self._lab_busy_until)), key=self._lab_busy_until.__getitem__)
            complete_minute = max(complete_minute, self._lab_busy_until[next_slot] + int(spec["turnaround_min"]))
            self._lab_busy_until[next_slot] = complete_minute

        self._pending_tests.append(
            PendingTest(patient_id=patient.patient_id, test_name=test_name, complete_minute=complete_minute)
        )
        self._state.total_cost += float(spec["cost"])
        if test_name in patient.required_tests:
            self._bump_reward("diagnostic_quality_delta", 0.8)
        else:
            patient.unnecessary_tests.add(test_name)
            self._bump_reward("resource_efficiency_delta", -0.5)
        self._events.append(f"Ordered {test_name} for {patient.patient_id}.")

    def _call_specialist(self, patient: InternalPatient, specialist: str) -> None:
        if specialist not in self._specialist_busy_until:
            self._events.append(f"Unknown specialist '{specialist}' requested for {patient.patient_id}.")
            return
        if self._specialist_busy_until[specialist] > self._state.shift_minute:
            self._events.append(f"Specialist '{specialist}' is busy for {patient.patient_id}.")
            return
        patient.consults_called.add(specialist)
        self._specialist_busy_until[specialist] = self._state.shift_minute + 30
        if specialist == patient.recommended_specialist:
            self._bump_reward("diagnostic_quality_delta", 0.4)
        self._events.append(f"Called {specialist} for {patient.patient_id}.")

    def _apply_disposition(self, patient: InternalPatient, disposition: str) -> None:
        outcome = "discharged" if disposition == "home" else "admitted"
        patient.location = "discharged" if disposition == "home" else "admitted"
        patient.terminal_outcome = outcome
        if disposition == patient.correct_disposition and patient.required_tests.issubset(patient.ordered_tests):
            patient.correct_disposition_hit = True
            self._bump_reward("disposition_quality_delta", 1.5)
            self._bump_reward("patient_safety_delta", 0.5)
        elif disposition == "home" and patient.true_esi <= 2:
            patient.unsafe_disposition = True
            self._bump_reward("patient_safety_delta", -4.0)
            self._bump_reward("disposition_quality_delta", -2.0)
        else:
            self._bump_reward("disposition_quality_delta", -0.8)
        self._events.append(f"{patient.patient_id} dispositioned to {disposition}.")

    def _advance_time(self) -> None:
        self._state.shift_minute += 5

    def _complete_tests(self) -> None:
        remaining: list[PendingTest] = []
        for pending in self._pending_tests:
            patient = self._patients[pending.patient_id]
            if pending.complete_minute <= self._state.shift_minute:
                patient.completed_tests[pending.test_name] = generate_test_result(patient.diagnosis, pending.test_name)
                self._events.append(f"Completed {pending.test_name} for {pending.patient_id}.")
                if pending.test_name in patient.required_tests:
                    self._bump_reward("diagnostic_quality_delta", 0.25)
            else:
                remaining.append(pending)
        self._pending_tests = remaining

    def _release_specialists(self) -> None:
        for specialist, busy_until in list(self._specialist_busy_until.items()):
            if busy_until <= self._state.shift_minute:
                self._specialist_busy_until[specialist] = 0

    def _spawn_arrivals(self) -> None:
        still_future: list[tuple[int, str | None]] = []
        for minute, forced in self._future_arrivals:
            if minute <= self._state.shift_minute:
                patient = self._create_patient(arrival_minute=minute, forced_diagnosis=forced)
                self._patients[patient.patient_id] = patient
                self._events.append(f"New arrival: {patient.patient_id} with {patient.chief_complaint}.")
            else:
                still_future.append((minute, forced))
        self._future_arrivals = still_future

    def _update_deterioration(self) -> None:
        for patient in self._patients.values():
            if not patient.is_active:
                continue
            if patient.trajectory == "stable":
                continue
            untreated = patient.location == "waiting" and not patient.required_tests.intersection(patient.ordered_tests)
            if not untreated and patient.location == "bed":
                continue

            worsen = 1
            if patient.trajectory == "rapid_decline":
                worsen = 2

            patient.vitals["hr"] = int(min(170, int(patient.vitals["hr"]) + (4 * worsen)))
            patient.vitals["rr"] = int(min(36, int(patient.vitals["rr"]) + (2 * worsen)))
            patient.vitals["sbp"] = int(max(70, int(patient.vitals["sbp"]) - (5 * worsen)))
            patient.vitals["o2_sat"] = int(max(75, int(patient.vitals["o2_sat"]) - worsen))
            if patient.diagnosis in {"sepsis", "meningitis"}:
                patient.vitals["temp_c"] = round(min(40.5, float(patient.vitals["temp_c"]) + 0.2), 1)
            if patient.diagnosis == "head_bleed":
                patient.vitals["gcs"] = max(3, int(patient.vitals["gcs"]) - 1)

            news2 = compute_news2(patient.vitals)
            if news2 >= HIGH_NEWS2_THRESHOLD:
                self._alerts.append(f"{patient.patient_id} has NEWS2 {news2}.")
                if patient.assigned_esi > 2 and patient.location == "waiting":
                    self._bump_reward("patient_safety_delta", -1.0)
                else:
                    self._bump_reward("patient_safety_delta", 0.1)

            if patient.true_esi <= 2 and patient.location == "waiting" and news2 >= CRITICAL_NEWS2_THRESHOLD:
                patient.terminal_outcome = "dead"
                patient.location = "discharged"
                self._bump_reward("patient_safety_delta", -6.0)
                self._bump_reward("timeliness_delta", -1.5)
                self._events.append(f"Critical deterioration: {patient.patient_id} died while untreated.")

    def _check_wait_breaches_and_lwbs(self) -> None:
        for patient in self._patients.values():
            if not patient.is_active:
                continue
            wait_time = self._state.shift_minute - patient.arrival_minute
            target = ESI_WAIT_TARGET_MIN[patient.true_esi]
            if (
                patient.true_esi <= 2
                and patient.location == "waiting"
                and wait_time > target
                and not patient.urgent_wait_breached
            ):
                patient.urgent_wait_breached = True
                self._bump_reward("timeliness_delta", -2.0)
                self._bump_reward("patient_safety_delta", -1.0)
                self._alerts.append(f"Urgent wait breach for {patient.patient_id}.")
            if patient.true_esi >= 4 and patient.location == "waiting" and wait_time >= 240:
                patient.terminal_outcome = "left_without_being_seen"
                patient.location = "discharged"
                self._bump_reward("timeliness_delta", -0.8)
                self._events.append(f"{patient.patient_id} left without being seen.")

    def _recount_state(self) -> None:
        self._state.total_seen = len(self._patients)
        self._state.total_discharged = sum(1 for p in self._patients.values() if p.terminal_outcome == "discharged")
        self._state.total_admitted = sum(1 for p in self._patients.values() if p.terminal_outcome == "admitted")
        deaths = sum(1 for p in self._patients.values() if p.terminal_outcome == "dead")
        unsafe = sum(1 for p in self._patients.values() if p.unsafe_disposition)
        self._state.lwbs_count = sum(
            1 for p in self._patients.values() if p.terminal_outcome == "left_without_being_seen"
        )
        self._state.adverse_events = deaths + unsafe + self._state.lwbs_count

    def _beds_in_use(self) -> int:
        return sum(1 for p in self._patients.values() if p.location == "bed" and p.is_active)

    def _ct_available_in_min(self) -> int:
        return max(0, self._ct_busy_until - self._state.shift_minute)

    def _lab_queue_length(self) -> int:
        return sum(1 for pending in self._pending_tests if TEST_CATALOG[pending.test_name]["resource"] == "lab")

    def _build_observation(self, done: bool) -> ERObservation:
        visible_patients = sorted(
            (self._to_patient_view(patient) for patient in self._patients.values() if patient.is_active),
            key=lambda patient: (-patient.news2_score, -patient.qsofa_score, -patient.wait_time_min),
        )
        resources = ResourceView(
            beds_available=max(0, self._task.beds_total - self._beds_in_use()),
            beds_total=self._task.beds_total,
            ct_available_in_min=self._ct_available_in_min(),
            lab_queue_length=self._lab_queue_length(),
            specialist_available={
                specialist: busy_until <= self._state.shift_minute
                for specialist, busy_until in self._specialist_busy_until.items()
            },
        )
        metadata: dict[str, Any] = {}
        self._finalize_step_reward()
        metadata["reward_breakdown"] = {
            key: round(value, 4) for key, value in self._reward_components.items()
        }
        if done and self._state.benchmark_score is not None:
            metadata["benchmark_score"] = self._state.benchmark_score
            metadata["benchmark_breakdown"] = self._state.benchmark_breakdown
        return ERObservation(
            done=done,
            reward=round(self._step_reward, 4),
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            shift_minute=self._state.shift_minute,
            patients=visible_patients,
            resources=resources,
            alerts=self._alerts,
            events=self._events,
            metadata=metadata,
        )

    def _to_patient_view(self, patient: InternalPatient) -> PatientView:
        return PatientView(
            patient_id=patient.patient_id,
            chief_complaint=patient.chief_complaint,
            age=patient.age,
            arrival_minute=patient.arrival_minute,
            wait_time_min=self._state.shift_minute - patient.arrival_minute,
            assigned_esi=patient.assigned_esi,
            location=patient.location,  # type: ignore[arg-type]
            vitals=VitalsView(**patient.vitals),
            pain_scale=patient.pain_scale,
            history=patient.history,
            completed_tests=patient.completed_tests,
            news2_score=compute_news2(patient.vitals),
            qsofa_score=compute_qsofa(patient.vitals),
        )
