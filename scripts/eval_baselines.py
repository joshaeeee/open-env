"""Run deterministic baseline policies for OpenER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

try:
    from open_er.models import ERAction, ERObservation, PatientCommand
    from open_er.server.environment import ERTriageEnvironment
    from open_er.server.tasks import OFFICIAL_TASKS
except ImportError:  # pragma: no cover - source tree fallback
    from models import ERAction, ERObservation, PatientCommand  # type: ignore
    from server.environment import ERTriageEnvironment  # type: ignore
    from server.tasks import OFFICIAL_TASKS  # type: ignore


FIXED_SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

CORE_TESTS = {
    "chest_pain": ["ecg", "troponin", "cbc", "bmp"],
    "shortness_of_breath": ["xray", "cbc", "bmp"],
    "abdominal_pain": ["cbc", "bmp", "ct_abdomen"],
    "fever": ["cbc", "bmp", "lactate"],
    "injury": ["xray"],
    "headache": ["cbc", "ct_head"],
}


def random_policy(observation: ERObservation) -> ERAction:
    return ERAction(commands=[])


def heuristic_policy(observation: ERObservation) -> ERAction:
    commands: list[PatientCommand] = []
    beds_available = observation.resources.beds_available
    sorted_patients = sorted(
        observation.patients,
        key=lambda patient: (-patient.news2_score, -patient.qsofa_score, -patient.wait_time_min),
    )
    for patient in sorted_patients:
        command = PatientCommand(patient_id=patient.patient_id)
        if beds_available > 0 and patient.location == "waiting" and (patient.news2_score >= 4 or patient.assigned_esi <= 2):
            command.assign_bed = True
            beds_available -= 1

        if patient.news2_score >= 7 or patient.qsofa_score >= 2:
            command.new_esi = 1
        elif patient.news2_score >= 5:
            command.new_esi = 2
        elif patient.news2_score >= 3:
            command.new_esi = 3

        completed_tests = patient.completed_tests
        completed_names = set(completed_tests)
        needed = CORE_TESTS.get(patient.chief_complaint, [])
        command.order_tests = [test for test in needed if test not in completed_names][:2]

        if patient.news2_score >= 6 or patient.qsofa_score >= 2:
            if patient.chief_complaint == "chest_pain":
                command.call_specialist = "cardiology"
            elif patient.chief_complaint == "abdominal_pain":
                command.call_specialist = "surgery"

        if patient.location == "bed":
            if patient.news2_score <= 2 and patient.assigned_esi >= 4 and len(completed_names) >= 1:
                command.disposition = "home"
            elif patient.news2_score >= 6 or patient.qsofa_score >= 2 or "ST elevation" in map(str, completed_tests.values()):
                command.disposition = "admit"

        if command.assign_bed or command.new_esi is not None or command.order_tests or command.disposition or command.call_specialist:
            commands.append(command)

    return ERAction(commands=commands)


POLICIES = {
    "heuristic": heuristic_policy,
    "random": random_policy,
}


def run_episode(task_id: str, seed: int, policy_name: str) -> dict[str, float | str | int]:
    env = ERTriageEnvironment()
    result = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    policy = POLICIES[policy_name]
    while not result.done:
        action = policy(result)
        result = env.step(action)
        total_reward += float(result.reward or 0.0)
    state = env.state
    return {
        "task_id": task_id,
        "seed": seed,
        "policy": policy_name,
        "score": float(state.benchmark_score or 0.0),
        "reward": round(total_reward, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), default="heuristic")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    seeds = FIXED_SEEDS[: args.episodes]
    outputs_dir = Path("outputs/evals")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | str | int]] = []
    for task_id in OFFICIAL_TASKS:
        task_runs = [run_episode(task_id, seed, args.policy) for seed in seeds]
        summaries.extend(task_runs)
        print(
            f"{task_id:22} mean_score={mean(run['score'] for run in task_runs):.3f} "
            f"mean_reward={mean(run['reward'] for run in task_runs):.3f} episodes={len(task_runs)}"
        )

    macro_score = mean(float(row["score"]) for row in summaries)
    macro_reward = mean(float(row["reward"]) for row in summaries)
    print(f"{'macro_average':22} mean_score={macro_score:.3f} mean_reward={macro_reward:.3f} episodes={len(summaries)}")
    print(f"seeds={seeds}")

    output_path = outputs_dir / f"{args.policy}_baseline.json"
    output_path.write_text(json.dumps({"policy": args.policy, "seeds": seeds, "runs": summaries}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
