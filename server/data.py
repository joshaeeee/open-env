"""Clinical-ish synthetic data and helper functions for OpenER."""

from __future__ import annotations

import copy
import random
from typing import Any


TEST_CATALOG: dict[str, dict[str, Any]] = {
    "ecg": {"cost": 25.0, "turnaround_min": 5, "resource": None},
    "troponin": {"cost": 50.0, "turnaround_min": 30, "resource": "lab"},
    "cbc": {"cost": 30.0, "turnaround_min": 20, "resource": "lab"},
    "bmp": {"cost": 40.0, "turnaround_min": 20, "resource": "lab"},
    "lactate": {"cost": 35.0, "turnaround_min": 15, "resource": "lab"},
    "blood_culture": {"cost": 65.0, "turnaround_min": 60, "resource": "lab"},
    "ct_head": {"cost": 1200.0, "turnaround_min": 45, "resource": "ct"},
    "ct_chest": {"cost": 1500.0, "turnaround_min": 45, "resource": "ct"},
    "ct_abdomen": {"cost": 1600.0, "turnaround_min": 45, "resource": "ct"},
    "xray": {"cost": 120.0, "turnaround_min": 15, "resource": "lab"},
    "urinalysis": {"cost": 25.0, "turnaround_min": 15, "resource": "lab"},
}


COMPLAINT_WEIGHTS = {
    "chest_pain": 0.20,
    "shortness_of_breath": 0.16,
    "abdominal_pain": 0.18,
    "fever": 0.16,
    "injury": 0.16,
    "headache": 0.14,
}


PATIENT_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "chest_pain": [
        {
            "name": "stemi",
            "probability": 0.18,
            "true_esi": 1,
            "trajectory": "rapid_decline",
            "correct_disposition": "admit",
            "required_tests": ["ecg", "troponin", "cbc", "bmp"],
            "specialist": "cardiology",
            "base_vitals": {"hr": 118, "sbp": 92, "dbp": 60, "rr": 24, "o2_sat": 92, "temp_c": 37.1, "gcs": 15},
        },
        {
            "name": "nstemi",
            "probability": 0.20,
            "true_esi": 2,
            "trajectory": "slow_decline",
            "correct_disposition": "admit",
            "required_tests": ["ecg", "troponin", "cbc", "bmp"],
            "specialist": "cardiology",
            "base_vitals": {"hr": 102, "sbp": 108, "dbp": 68, "rr": 20, "o2_sat": 95, "temp_c": 36.9, "gcs": 15},
        },
        {
            "name": "anxiety",
            "probability": 0.62,
            "true_esi": 4,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["ecg"],
            "specialist": None,
            "base_vitals": {"hr": 96, "sbp": 128, "dbp": 80, "rr": 18, "o2_sat": 98, "temp_c": 36.8, "gcs": 15},
        },
    ],
    "shortness_of_breath": [
        {
            "name": "pe",
            "probability": 0.22,
            "true_esi": 1,
            "trajectory": "rapid_decline",
            "correct_disposition": "admit",
            "required_tests": ["cbc", "bmp", "ct_chest"],
            "specialist": "pulmonology",
            "base_vitals": {"hr": 122, "sbp": 94, "dbp": 58, "rr": 28, "o2_sat": 88, "temp_c": 37.0, "gcs": 15},
        },
        {
            "name": "chf",
            "probability": 0.28,
            "true_esi": 2,
            "trajectory": "slow_decline",
            "correct_disposition": "admit",
            "required_tests": ["bmp", "cbc", "xray"],
            "specialist": "cardiology",
            "base_vitals": {"hr": 108, "sbp": 110, "dbp": 70, "rr": 24, "o2_sat": 90, "temp_c": 36.9, "gcs": 15},
        },
        {
            "name": "asthma",
            "probability": 0.50,
            "true_esi": 3,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["xray"],
            "specialist": None,
            "base_vitals": {"hr": 96, "sbp": 122, "dbp": 76, "rr": 22, "o2_sat": 95, "temp_c": 36.8, "gcs": 15},
        },
    ],
    "abdominal_pain": [
        {
            "name": "appendicitis",
            "probability": 0.30,
            "true_esi": 2,
            "trajectory": "slow_decline",
            "correct_disposition": "admit",
            "required_tests": ["cbc", "bmp", "ct_abdomen"],
            "specialist": "surgery",
            "base_vitals": {"hr": 104, "sbp": 108, "dbp": 70, "rr": 20, "o2_sat": 97, "temp_c": 38.0, "gcs": 15},
        },
        {
            "name": "biliary_colic",
            "probability": 0.24,
            "true_esi": 3,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["cbc", "bmp"],
            "specialist": None,
            "base_vitals": {"hr": 90, "sbp": 124, "dbp": 78, "rr": 16, "o2_sat": 98, "temp_c": 36.9, "gcs": 15},
        },
        {
            "name": "gastroenteritis",
            "probability": 0.46,
            "true_esi": 4,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["cbc"],
            "specialist": None,
            "base_vitals": {"hr": 86, "sbp": 118, "dbp": 74, "rr": 16, "o2_sat": 99, "temp_c": 37.2, "gcs": 15},
        },
    ],
    "fever": [
        {
            "name": "sepsis",
            "probability": 0.28,
            "true_esi": 1,
            "trajectory": "rapid_decline",
            "correct_disposition": "admit",
            "required_tests": ["cbc", "bmp", "lactate", "blood_culture"],
            "specialist": "critical_care",
            "base_vitals": {"hr": 126, "sbp": 88, "dbp": 54, "rr": 28, "o2_sat": 91, "temp_c": 39.4, "gcs": 14},
        },
        {
            "name": "pyelonephritis",
            "probability": 0.22,
            "true_esi": 3,
            "trajectory": "slow_decline",
            "correct_disposition": "admit",
            "required_tests": ["cbc", "bmp", "urinalysis"],
            "specialist": None,
            "base_vitals": {"hr": 104, "sbp": 112, "dbp": 68, "rr": 20, "o2_sat": 97, "temp_c": 38.7, "gcs": 15},
        },
        {
            "name": "viral_syndrome",
            "probability": 0.50,
            "true_esi": 4,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["cbc"],
            "specialist": None,
            "base_vitals": {"hr": 92, "sbp": 120, "dbp": 76, "rr": 18, "o2_sat": 98, "temp_c": 38.1, "gcs": 15},
        },
    ],
    "injury": [
        {
            "name": "head_bleed",
            "probability": 0.20,
            "true_esi": 1,
            "trajectory": "rapid_decline",
            "correct_disposition": "admit",
            "required_tests": ["ct_head", "cbc"],
            "specialist": "neurosurgery",
            "base_vitals": {"hr": 58, "sbp": 168, "dbp": 96, "rr": 16, "o2_sat": 97, "temp_c": 36.6, "gcs": 12},
        },
        {
            "name": "fracture",
            "probability": 0.44,
            "true_esi": 3,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["xray"],
            "specialist": None,
            "base_vitals": {"hr": 92, "sbp": 132, "dbp": 82, "rr": 18, "o2_sat": 98, "temp_c": 36.7, "gcs": 15},
        },
        {
            "name": "sprain",
            "probability": 0.36,
            "true_esi": 4,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": ["xray"],
            "specialist": None,
            "base_vitals": {"hr": 86, "sbp": 126, "dbp": 80, "rr": 16, "o2_sat": 99, "temp_c": 36.6, "gcs": 15},
        },
    ],
    "headache": [
        {
            "name": "migraine",
            "probability": 0.60,
            "true_esi": 4,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": [],
            "specialist": None,
            "base_vitals": {"hr": 84, "sbp": 122, "dbp": 78, "rr": 16, "o2_sat": 99, "temp_c": 36.7, "gcs": 15},
        },
        {
            "name": "meningitis",
            "probability": 0.10,
            "true_esi": 2,
            "trajectory": "slow_decline",
            "correct_disposition": "admit",
            "required_tests": ["cbc", "bmp", "ct_head"],
            "specialist": "neurology",
            "base_vitals": {"hr": 108, "sbp": 110, "dbp": 68, "rr": 22, "o2_sat": 97, "temp_c": 39.0, "gcs": 14},
        },
        {
            "name": "tension_headache",
            "probability": 0.30,
            "true_esi": 5,
            "trajectory": "stable",
            "correct_disposition": "home",
            "required_tests": [],
            "specialist": None,
            "base_vitals": {"hr": 78, "sbp": 120, "dbp": 76, "rr": 15, "o2_sat": 99, "temp_c": 36.7, "gcs": 15},
        },
    ],
}


SPECIALISTS = [
    "cardiology",
    "surgery",
    "critical_care",
    "neurosurgery",
    "neurology",
    "pulmonology",
]


ESI_WAIT_TARGET_MIN = {
    1: 0,
    2: 15,
    3: 30,
    4: 90,
    5: 120,
}

HIGH_NEWS2_THRESHOLD = 7
CRITICAL_NEWS2_THRESHOLD = 9
HIGH_QSOFA_THRESHOLD = 2


def _weighted_choice(entries: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    total = sum(entry["probability"] for entry in entries)
    needle = rng.random() * total
    cursor = 0.0
    for entry in entries:
        cursor += entry["probability"]
        if cursor >= needle:
            return copy.deepcopy(entry)
    return copy.deepcopy(entries[-1])


def sample_complaint(rng: random.Random) -> str:
    complaints = list(COMPLAINT_WEIGHTS)
    weights = [COMPLAINT_WEIGHTS[c] for c in complaints]
    return rng.choices(complaints, weights=weights, k=1)[0]


def sample_patient_blueprint(
    rng: random.Random,
    forced_diagnosis: str | None = None,
) -> dict[str, Any]:
    complaint = sample_complaint(rng)
    if forced_diagnosis is None:
        diagnosis = _weighted_choice(PATIENT_LIBRARY[complaint], rng)
    else:
        for candidate_complaint, entries in PATIENT_LIBRARY.items():
            for entry in entries:
                if entry["name"] == forced_diagnosis:
                    complaint = candidate_complaint
                    diagnosis = copy.deepcopy(entry)
                    break
            else:
                continue
            break
        else:
            raise ValueError(f"Unknown forced diagnosis: {forced_diagnosis}")

    base_vitals = copy.deepcopy(diagnosis["base_vitals"])
    base_vitals["hr"] = int(base_vitals["hr"] + rng.randint(-6, 6))
    base_vitals["sbp"] = int(base_vitals["sbp"] + rng.randint(-5, 5))
    base_vitals["dbp"] = int(base_vitals["dbp"] + rng.randint(-4, 4))
    base_vitals["rr"] = int(base_vitals["rr"] + rng.randint(-2, 2))
    base_vitals["o2_sat"] = max(78, min(100, base_vitals["o2_sat"] + rng.randint(-2, 2)))
    base_vitals["temp_c"] = round(base_vitals["temp_c"] + rng.uniform(-0.2, 0.2), 1)
    base_vitals["gcs"] = max(3, min(15, base_vitals["gcs"] + rng.randint(-1, 1)))

    return {
        "chief_complaint": complaint,
        "diagnosis": diagnosis["name"],
        "true_esi": diagnosis["true_esi"],
        "trajectory": diagnosis["trajectory"],
        "correct_disposition": diagnosis["correct_disposition"],
        "required_tests": list(diagnosis["required_tests"]),
        "specialist": diagnosis["specialist"],
        "vitals": base_vitals,
        "age": rng.randint(22, 83),
        "pain_scale": rng.randint(2, 9),
        "history": sorted(rng.sample(["diabetes", "hypertension", "asthma", "cad", "none"], k=2)),
    }


def generate_test_result(diagnosis: str, test_name: str) -> str | int | float:
    mappings: dict[str, dict[str, str | int | float]] = {
        "stemi": {"ecg": "ST elevation", "troponin": 5.6, "cbc": "mild leukocytosis", "bmp": "normal"},
        "nstemi": {"ecg": "nonspecific changes", "troponin": 1.9, "cbc": "normal", "bmp": "normal"},
        "anxiety": {"ecg": "normal sinus rhythm"},
        "appendicitis": {"cbc": "leukocytosis", "bmp": "normal", "ct_abdomen": "inflamed appendix"},
        "gastroenteritis": {"cbc": "normal"},
        "biliary_colic": {"cbc": "normal", "bmp": "normal"},
        "sepsis": {"cbc": "marked leukocytosis", "bmp": "acute kidney injury", "lactate": 4.7, "blood_culture": "positive"},
        "pyelonephritis": {"cbc": "leukocytosis", "bmp": "mild aki", "urinalysis": "nitrites positive"},
        "viral_syndrome": {"cbc": "normal"},
        "head_bleed": {"ct_head": "intracranial bleed", "cbc": "normal"},
        "fracture": {"xray": "fracture present"},
        "sprain": {"xray": "no fracture"},
        "migraine": {},
        "meningitis": {"cbc": "leukocytosis", "bmp": "normal", "ct_head": "no acute bleed"},
        "tension_headache": {},
        "pe": {"cbc": "normal", "bmp": "normal", "ct_chest": "pulmonary embolism"},
        "chf": {"bmp": "elevated bnp surrogate", "cbc": "normal", "xray": "pulmonary edema"},
        "asthma": {"xray": "hyperinflation"},
    }
    return mappings.get(diagnosis, {}).get(test_name, "normal")


def compute_news2(vitals: dict[str, int | float]) -> int:
    score = 0
    rr = int(vitals["rr"])
    o2 = int(vitals["o2_sat"])
    sbp = int(vitals["sbp"])
    hr = int(vitals["hr"])
    temp = float(vitals["temp_c"])
    gcs = int(vitals["gcs"])

    if rr <= 8 or rr >= 25:
        score += 3
    elif rr >= 21:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    if o2 <= 91:
        score += 3
    elif o2 <= 93:
        score += 2
    elif o2 <= 95:
        score += 1

    if sbp <= 90 or sbp >= 220:
        score += 3
    elif sbp <= 100:
        score += 2
    elif sbp <= 110:
        score += 1

    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130 or 41 <= hr <= 50:
        score += 2
    elif 91 <= hr <= 110:
        score += 1

    if temp <= 35.0:
        score += 3
    elif temp >= 39.1:
        score += 2
    elif 38.1 <= temp <= 39.0 or 35.1 <= temp <= 36.0:
        score += 1

    if gcs < 15:
        score += 3

    return score


def compute_qsofa(vitals: dict[str, int | float]) -> int:
    score = 0
    if int(vitals["rr"]) >= 22:
        score += 1
    if int(vitals["sbp"]) <= 100:
        score += 1
    if int(vitals["gcs"]) < 15:
        score += 1
    return score


def arrival_rate(profile: str, step_index: int) -> int:
    """Return number of arrivals for the next step."""
    pattern = {
        "low": [0, 1, 0, 0, 1],
        "medium": [1, 0, 1, 1, 0],
        "high": [1, 1, 2, 1, 1],
    }
    seq = pattern[profile]
    return seq[step_index % len(seq)]
