"""
Inference runner for OpenER.

MANDATORY
- Ensure the following environment variables are available before submitting:
    API_BASE_URL: Base URL for the OpenAI-compatible inference endpoint.
    MODEL_NAME: Model identifier used for chat completions.
    HF_TOKEN: API token for the inference endpoint.
    LOCAL_IMAGE_NAME: Optional local Docker image name when launching the env from Docker.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- This script must be named `inference.py` and live at the project root.
- All LLM calls use the OpenAI Python client.

STDOUT FORMAT
- The script emits exactly these line types to stdout, in order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
BENCHMARK = "open_er"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

OPEN_ER_REPO_ID = os.getenv("OPEN_ER_REPO_ID", "joshaeeee/open-er")
OPEN_ER_BASE_URL = os.getenv("OPEN_ER_BASE_URL")
TASK_NAME = os.getenv("OPEN_ER_TASK", "easy_single_critical")
SEED = int(os.getenv("OPEN_ER_SEED", "11"))
MAX_STEPS = int(os.getenv("OPEN_ER_MAX_STEPS", "96"))
TEMPERATURE = float(os.getenv("OPEN_ER_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("OPEN_ER_MAX_TOKENS", "900"))
REQUEST_TIMEOUT_S = float(os.getenv("OPEN_ER_REQUEST_TIMEOUT_S", "60"))


def _load_source_package() -> None:
    if "open_er" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "open_er",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load local open_er package")

    module = importlib.util.module_from_spec(spec)
    sys.modules["open_er"] = module
    spec.loader.exec_module(module)


try:
    from open_er.client import OpenEREnv
    from open_er.models import ERAction, ERObservation, PatientCommand
except ImportError:
    _load_source_package()
    from open_er.client import OpenEREnv
    from open_er.models import ERAction, ERObservation, PatientCommand


VALID_TESTS = [
    "ecg",
    "troponin",
    "cbc",
    "bmp",
    "lactate",
    "blood_culture",
    "ct_head",
    "ct_chest",
    "ct_abdomen",
    "xray",
    "urinalysis",
]

VALID_SPECIALISTS = [
    "cardiology",
    "surgery",
    "critical_care",
    "neurosurgery",
    "neurology",
    "pulmonology",
]

DEFAULT_TESTS_BY_COMPLAINT = {
    "chest_pain": ["ecg", "troponin", "cbc", "bmp"],
    "shortness_of_breath": ["xray", "cbc", "bmp"],
    "abdominal_pain": ["cbc", "bmp", "ct_abdomen"],
    "fever": ["cbc", "bmp", "lactate"],
    "injury": ["xray"],
    "headache": ["cbc", "ct_head"],
}

SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are optimizing actions for the OpenER emergency department triage benchmark.

    Primary objectives, in order:
    1. Prevent deaths, unsafe discharges, and severe delays for critical patients.
    2. Put the sickest waiting patients into beds quickly.
    3. Order only useful tests from this valid set: {", ".join(VALID_TESTS)}.
    4. Call specialists only from this valid set: {", ".join(VALID_SPECIALISTS)}.
    5. Choose the correct disposition only when visible evidence supports it.

    You do NOT know the hidden diagnosis. Use only visible vitals, complaint, wait time,
    NEWS2, qSOFA, completed tests, alerts, and events.

    Return JSON only with this shape:
    {{
      "commands": [
        {{
          "patient_id": "pt_001_0",
          "assign_bed": true,
          "new_esi": 1,
          "order_tests": ["ecg", "troponin"],
          "disposition": "admit",
          "call_specialist": "cardiology"
        }}
      ]
    }}

    Rules:
    - Omit fields that are not needed.
    - Use at most one command object per patient.
    - Do not include comments, markdown, or prose outside JSON.
    - Do not discharge high-risk patients.
    - Avoid duplicate or already-completed tests.
    - Prefer no-op over speculative low-value actions.
    """
).strip()


def _single_line(value: str | None) -> str:
    if value is None:
        return "null"
    collapsed = " ".join(str(value).split())
    return collapsed if collapsed else "null"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_single_line(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def _first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model response")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Environment at {base_url} did not become ready within {timeout_s:.1f}s")


@dataclass
class ManagedDockerContainer:
    image_name: str
    host_port: int
    container_id: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    @classmethod
    def start(cls, image_name: str) -> "ManagedDockerContainer":
        try:
            subprocess.run(
                ["docker", "version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("Docker is required when LOCAL_IMAGE_NAME is set") from exc

        host_port = _find_free_port()
        command = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{host_port}:7860",
            image_name,
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to start Docker image {image_name}: {exc.stderr.strip()}") from exc

        container_id = result.stdout.strip()
        managed = cls(image_name=image_name, host_port=host_port, container_id=container_id)
        _wait_for_health(managed.base_url)
        return managed

    def stop(self) -> None:
        subprocess.run(
            ["docker", "stop", self.container_id],
            capture_output=True,
            text=True,
            check=False,
        )


def _severity_signal(patient: Any) -> str:
    signals: list[str] = []
    if patient.news2_score >= 7:
        signals.append("high_news2")
    if patient.qsofa_score >= 2:
        signals.append("high_qsofa")
    if patient.vitals.o2_sat <= 92:
        signals.append("hypoxic")
    if patient.vitals.sbp <= 95:
        signals.append("hypotensive")
    if patient.vitals.gcs <= 13:
        signals.append("low_gcs")
    return ",".join(signals) if signals else "none"


def _format_completed_tests(patient: Any) -> str:
    if not patient.completed_tests:
        return "none"
    chunks = [f"{name}={value}" for name, value in patient.completed_tests.items()]
    return "; ".join(chunks)


def _format_observation(observation: ERObservation) -> str:
    patient_lines = []
    for patient in observation.patients:
        patient_lines.append(
            (
                f"- {patient.patient_id}: complaint={patient.chief_complaint} location={patient.location} "
                f"wait={patient.wait_time_min}m esi={patient.assigned_esi} news2={patient.news2_score} "
                f"qsofa={patient.qsofa_score} age={patient.age} pain={patient.pain_scale} "
                f"vitals=(hr={patient.vitals.hr}, sbp={patient.vitals.sbp}, rr={patient.vitals.rr}, "
                f"o2={patient.vitals.o2_sat}, temp={patient.vitals.temp_c}, gcs={patient.vitals.gcs}) "
                f"signals={_severity_signal(patient)} completed_tests={_format_completed_tests(patient)}"
            )
        )

    alerts = observation.alerts or ["none"]
    events = observation.events or ["none"]
    resources = observation.resources
    return textwrap.dedent(
        f"""
        Task: {observation.task_id}
        Difficulty: {observation.difficulty}
        Shift minute: {observation.shift_minute}
        Resources:
        - beds_available={resources.beds_available}/{resources.beds_total}
        - ct_available_in_min={resources.ct_available_in_min}
        - lab_queue_length={resources.lab_queue_length}
        - specialists={json.dumps(resources.specialist_available, sort_keys=True)}
        Alerts:
        {os.linesep.join(f"- {alert}" for alert in alerts)}
        Events:
        {os.linesep.join(f"- {event}" for event in events)}
        Patients:
        {os.linesep.join(patient_lines) if patient_lines else "- none"}
        """
    ).strip()


def _heuristic_command_for_patient(patient: Any, beds_available: int, specialist_available: dict[str, bool]) -> tuple[dict[str, Any], int]:
    command: dict[str, Any] = {"patient_id": patient.patient_id}

    if (
        beds_available > 0
        and patient.location == "waiting"
        and (
            patient.news2_score >= 4
            or patient.qsofa_score >= 2
            or patient.assigned_esi <= 2
            or patient.vitals.o2_sat <= 92
            or patient.vitals.sbp <= 95
            or patient.vitals.gcs <= 13
            or patient.wait_time_min >= 30 and patient.assigned_esi <= 3
        )
    ):
        command["assign_bed"] = True
        beds_available -= 1

    if patient.news2_score >= 7 or patient.qsofa_score >= 2 or patient.vitals.gcs <= 12:
        command["new_esi"] = 1
    elif patient.news2_score >= 5 or patient.vitals.o2_sat <= 92 or patient.vitals.sbp <= 95:
        command["new_esi"] = 2
    elif patient.news2_score >= 3:
        command["new_esi"] = 3

    completed_names = set(patient.completed_tests)
    suggested_tests = DEFAULT_TESTS_BY_COMPLAINT.get(patient.chief_complaint, [])

    if patient.chief_complaint == "shortness_of_breath" and (
        patient.news2_score >= 7 or patient.qsofa_score >= 2 or patient.vitals.o2_sat <= 90
    ):
        suggested_tests = suggested_tests + ["ct_chest"]
    if patient.chief_complaint == "fever" and (
        patient.qsofa_score >= 2 or patient.news2_score >= 7 or patient.vitals.sbp <= 95
    ):
        suggested_tests = suggested_tests + ["blood_culture"]

    command["order_tests"] = [test for test in suggested_tests if test not in completed_names][:4]

    if patient.news2_score >= 6 or patient.qsofa_score >= 2 or patient.vitals.gcs <= 12:
        if patient.chief_complaint == "chest_pain" and specialist_available.get("cardiology", False):
            command["call_specialist"] = "cardiology"
        elif patient.chief_complaint == "abdominal_pain" and specialist_available.get("surgery", False):
            command["call_specialist"] = "surgery"
        elif patient.chief_complaint == "fever" and specialist_available.get("critical_care", False):
            command["call_specialist"] = "critical_care"
        elif patient.chief_complaint == "injury" and specialist_available.get("neurosurgery", False):
            command["call_specialist"] = "neurosurgery"
        elif patient.chief_complaint == "headache" and specialist_available.get("neurology", False):
            command["call_specialist"] = "neurology"
        elif patient.chief_complaint == "shortness_of_breath" and specialist_available.get("pulmonology", False):
            command["call_specialist"] = "pulmonology"

    if patient.location == "bed":
        test_values = {str(value) for value in patient.completed_tests.values()}
        if (
            patient.news2_score >= 6
            or patient.qsofa_score >= 2
            or "ST elevation" in test_values
            or patient.vitals.gcs <= 12
            or patient.vitals.o2_sat <= 90
        ):
            command["disposition"] = "admit"
        elif (
            patient.assigned_esi >= 4
            and patient.news2_score <= 2
            and patient.qsofa_score == 0
            and (
                patient.completed_tests
                or patient.chief_complaint == "headache"
            )
        ):
            command["disposition"] = "home"

    return command, beds_available


def heuristic_action(observation: ERObservation) -> ERAction:
    commands: list[PatientCommand] = []
    beds_available = observation.resources.beds_available
    specialist_available = observation.resources.specialist_available
    sorted_patients = sorted(
        observation.patients,
        key=lambda patient: (-patient.news2_score, -patient.qsofa_score, -patient.wait_time_min),
    )

    for patient in sorted_patients:
        raw_command, beds_available = _heuristic_command_for_patient(patient, beds_available, specialist_available)
        if any(
            raw_command.get(key)
            for key in ("assign_bed", "new_esi", "order_tests", "disposition", "call_specialist")
        ):
            commands.append(PatientCommand.model_validate(raw_command))

    return ERAction(commands=commands)


def _normalize_model_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"commands": payload}
    if not isinstance(payload, dict):
        raise ValueError("Model response must decode to an object or a command list")
    if "commands" in payload:
        return payload
    return {"commands": payload.get("actions", [])}


def sanitize_action(raw_payload: Any, observation: ERObservation) -> ERAction:
    normalized = _normalize_model_payload(raw_payload)
    visible_patients = {patient.patient_id: patient for patient in observation.patients}
    commands: list[PatientCommand] = []
    seen: set[str] = set()

    for item in normalized.get("commands", []):
        if not isinstance(item, dict):
            continue

        patient_id = item.get("patient_id") or item.get("patient") or item.get("id")
        if patient_id not in visible_patients or patient_id in seen:
            continue
        seen.add(patient_id)

        new_esi = item.get("new_esi", item.get("esi", item.get("triage_level")))
        if new_esi is not None:
            try:
                new_esi = max(1, min(5, int(new_esi)))
            except (TypeError, ValueError):
                new_esi = None

        order_tests = item.get("order_tests", item.get("tests", []))
        if not isinstance(order_tests, list):
            order_tests = []
        clean_tests: list[str] = []
        completed = set(visible_patients[patient_id].completed_tests)
        for test_name in order_tests:
            if not isinstance(test_name, str):
                continue
            test_name = test_name.strip()
            if test_name in VALID_TESTS and test_name not in completed and test_name not in clean_tests:
                clean_tests.append(test_name)

        call_specialist = item.get("call_specialist", item.get("specialist", item.get("consult")))
        if call_specialist is not None:
            call_specialist = str(call_specialist).strip()
            if call_specialist not in VALID_SPECIALISTS:
                call_specialist = None

        disposition = item.get("disposition")
        if disposition is not None:
            disposition = str(disposition).strip().lower()
            if disposition not in {"home", "admit"}:
                disposition = None

        assign_bed = bool(item.get("assign_bed", item.get("bed", False)))

        command_payload = {
            "patient_id": patient_id,
            "assign_bed": assign_bed,
            "new_esi": new_esi,
            "order_tests": clean_tests,
            "disposition": disposition,
            "call_specialist": call_specialist,
        }
        command = PatientCommand.model_validate(command_payload)
        if command.assign_bed or command.new_esi is not None or command.order_tests or command.disposition or command.call_specialist:
            commands.append(command)

    return ERAction(commands=commands)


def action_to_log_string(action: ERAction) -> str:
    payload = {"commands": action.model_dump(exclude={"metadata"}).get("commands", [])}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def build_user_prompt(observation: ERObservation, heuristic: ERAction) -> str:
    return textwrap.dedent(
        f"""
        Produce the next OpenER action for exactly this observation.

        Current observation:
        {_format_observation(observation)}

        Baseline heuristic proposal:
        {json.dumps({"commands": heuristic.model_dump(exclude={"metadata"}).get("commands", [])}, indent=2, sort_keys=True)}

        Improve on the baseline if you can, but stay conservative and valid.
        Return JSON only.
        """
    ).strip()


def request_model_action(client: OpenAI, observation: ERObservation) -> tuple[ERAction, str | None]:
    heuristic = heuristic_action(observation)
    user_prompt = build_user_prompt(observation, heuristic)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = _first_json_object(content)
        action = sanitize_action(parsed, observation)
        return action, None
    except Exception as exc:
        return heuristic, f"model_fallback:{_single_line(str(exc))}"


async def connect_env() -> tuple[Any, ManagedDockerContainer | None]:
    if OPEN_ER_BASE_URL:
        env = OpenEREnv(base_url=OPEN_ER_BASE_URL)
        await env.connect()
        return env, None

    if LOCAL_IMAGE_NAME:
        managed = ManagedDockerContainer.start(LOCAL_IMAGE_NAME)
        env = OpenEREnv(base_url=managed.base_url)
        await env.connect()
        return env, managed

    env = await OpenEREnv.from_env(
        OPEN_ER_REPO_ID,
        use_docker=False,
        project_path=str(ROOT),
    )
    return env, None


def benchmark_score_from_result(result: Any, state: Any | None) -> float:
    metadata = getattr(result.observation, "metadata", {}) or {}
    score = metadata.get("benchmark_score")
    if score is None and state is not None:
        score = getattr(state, "benchmark_score", None)
    if score is None:
        return 0.0
    return max(0.0, min(1.0, float(score)))


async def main() -> None:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("HF_TOKEN is required for inference")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key, timeout=REQUEST_TIMEOUT_S)

    env = None
    managed_container: ManagedDockerContainer | None = None
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    final_state = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env, managed_container = await connect_env()
        result = await env.reset(task_id=TASK_NAME, seed=SEED)

        while not result.done and steps_taken < MAX_STEPS:
            action, action_error = request_model_action(client, result.observation)
            result = await env.step(action)
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken += 1
            log_step(
                step=steps_taken,
                action=action_to_log_string(action),
                reward=reward,
                done=bool(result.done),
                error=action_error,
            )
            if result.done:
                break

        try:
            final_state = await env.state()
        except Exception:
            final_state = None

        score = benchmark_score_from_result(result, final_state)
        success = bool(result.done) and 0.0 <= score <= 1.0
    finally:
        try:
            if env is not None:
                await env.close()
        finally:
            if managed_container is not None:
                managed_container.stop()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
