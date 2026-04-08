"""Deterministic benchmark grading for OpenER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .tasks import TaskConfig


@dataclass
class BenchmarkResult:
    score: float
    breakdown: dict[str, float]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def grade_episode(task: TaskConfig, patients: Iterable[object]) -> BenchmarkResult:
    patient_list = list(patients)
    total = max(1, len(patient_list))
    critical = [p for p in patient_list if getattr(p, "true_esi", 5) <= 2]
    critical_total = max(1, len(critical))

    deaths = sum(1 for p in patient_list if getattr(p, "terminal_outcome", None) == "dead")
    lwbs = sum(
        1 for p in patient_list if getattr(p, "terminal_outcome", None) == "left_without_being_seen"
    )
    unsafe = sum(1 for p in patient_list if getattr(p, "unsafe_disposition", False))
    urgent_wait_breaches = sum(1 for p in patient_list if getattr(p, "urgent_wait_breached", False))

    required_tests_total = sum(len(getattr(p, "required_tests", set())) for p in patient_list)
    required_tests_done = sum(
        len(set(getattr(p, "required_tests", set())) & set(getattr(p, "ordered_tests", set())))
        for p in patient_list
    )
    total_tests = sum(len(getattr(p, "ordered_tests", set())) for p in patient_list)
    unnecessary_tests = sum(len(getattr(p, "unnecessary_tests", set())) for p in patient_list)

    disposition_total = max(1, sum(1 for p in patient_list if getattr(p, "terminal_outcome", None) in {"admitted", "discharged"}))
    correct_dispositions = sum(1 for p in patient_list if getattr(p, "correct_disposition_hit", False))

    critical_stabilized = sum(
        1
        for p in critical
        if getattr(p, "terminal_outcome", None) not in {"dead", "discharged"}
    )

    patient_safety = _clip01(
        1.0 - ((2.0 * deaths) + unsafe + (0.5 * urgent_wait_breaches)) / (2.5 * critical_total)
    )
    timeliness = _clip01(1.0 - (urgent_wait_breaches + (0.5 * lwbs)) / max(1, total))
    diagnostic_quality = _clip01(
        0.6 * (required_tests_done / max(1, required_tests_total))
        + 0.4 * (correct_dispositions / disposition_total)
    )
    resource_stewardship = _clip01(
        1.0 - unnecessary_tests / max(1, total_tests + 1)
    )

    breakdown = {
        "patient_safety": patient_safety,
        "timeliness": timeliness,
        "diagnostic_quality": diagnostic_quality,
        "resource_stewardship": resource_stewardship,
        "critical_stabilization": _clip01(critical_stabilized / critical_total),
    }

    score = 0.0
    for key, weight in task.grader_weights.items():
        score += breakdown[key] * weight

    return BenchmarkResult(score=_clip01(score), breakdown=breakdown)
