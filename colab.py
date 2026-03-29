"""Colab-friendly TRL/OpenEnv training helpers for OpenER.

This module packages the full BrowserGym-style training flow:

1. system prompt
2. environment configuration
3. rollout helpers
4. reward functions
5. dataset creation
6. GRPO config creation
7. trainer creation and training entrypoint

The heavy TRL/datasets imports are intentionally lazy so this module can be
imported locally without requiring the full Colab training stack.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

from .client import OpenEREnv
from .models import ERAction, ERObservation
from .server.data import SPECIALISTS, TEST_CATALOG
from .server.tasks import OFFICIAL_TASKS


DEFAULT_BASE_URL = "https://joshaeeee-open-er.hf.space"
DEFAULT_DATASET_PROMPT = "Manage the emergency department safely and efficiently."

SYSTEM_PROMPT = f"""You are an expert emergency department triage and throughput agent operating a simulated ER shift.

Your job is to make safe, high-quality, resource-aware decisions for multiple concurrent patients.

PRIMARY OBJECTIVE
- Protect patient safety first.
- Then improve timeliness, diagnostic quality, and resource stewardship.

IMPORTANT SAFETY PRINCIPLES
- Undertriage is dangerous. If a patient looks unstable, escalate acuity rather than minimizing risk.
- Unsafe discharge is heavily penalized. Do not discharge unstable or insufficiently worked-up patients.
- Critical visible deterioration matters more than throughput.
- Use only the visible information in the observation. Hidden diagnoses are not available to you.

ENVIRONMENT FACTS
- One environment step represents a 5-minute turn in the ER.
- You may act on multiple patients in a single step.
- Beds, CT, labs, and specialists are resource constrained.
- Some patients deteriorate if they wait too long or are mishandled.
- Official tasks: {", ".join(OFFICIAL_TASKS)}

AVAILABLE ACTION FIELDS
- patient_id: the visible patient identifier
- assign_bed: true or false
- new_esi: integer 1-5 or null
- order_tests: list of test names
- disposition: "home", "admit", or null
- call_specialist: specialist name or null

VALID TEST NAMES
- {", ".join(sorted(TEST_CATALOG))}

VALID SPECIALISTS
- {", ".join(SPECIALISTS)}

OUTPUT FORMAT
- Output JSON only.
- Do not include markdown, explanations, analysis, or prose.
- The JSON must match this schema exactly:
{{
  "commands": [
    {{
      "patient_id": "pt_001_0",
      "assign_bed": false,
      "new_esi": null,
      "order_tests": [],
      "disposition": null,
      "call_specialist": null
    }}
  ]
}}

ACTION QUALITY GUIDELINES
- Assign beds to the most unstable or highest-acuity waiting patients first.
- Increase ESI urgency when NEWS2/qSOFA and vitals suggest instability.
- Order complaint-appropriate tests. Avoid wasteful shotgun testing.
- Call specialists when complaint pattern and severity make that consult plausible.
- Admit unstable or high-risk patients; discharge only stable low-acuity patients with enough workup.
- If no action is justified, return {{"commands": []}}.
"""


@dataclass
class OpenERTRLConfig:
    """Configuration for TRL/OpenEnv rollouts against OpenER."""

    base_url: str = DEFAULT_BASE_URL
    task_id: str = "easy_single_critical"
    seed_start: int = 11
    max_patients_in_prompt: int = 12
    max_episode_steps: int = 8
    dense_reward_scale: float = 10.0
    dense_reward_weight: float = 0.25
    benchmark_weight: float = 1.0
    system_prompt: str = SYSTEM_PROMPT
    dataset_prompt: str = DEFAULT_DATASET_PROMPT


def prompt_to_text(prompt: Any) -> str:
    """Normalize TRL prompt payloads into plain text."""

    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(prompt)


def render_observation(observation: ERObservation, max_patients: int = 12) -> str:
    """Render a compact textual summary of an OpenER observation."""

    lines = [
        f"task_id: {observation.task_id}",
        f"difficulty: {observation.difficulty}",
        f"shift_minute: {observation.shift_minute}",
        (
            "resources: "
            f"beds_available={observation.resources.beds_available}/{observation.resources.beds_total}, "
            f"ct_available_in_min={observation.resources.ct_available_in_min}, "
            f"lab_queue_length={observation.resources.lab_queue_length}"
        ),
    ]

    if observation.alerts:
        lines.append(f"alerts: {', '.join(observation.alerts)}")

    if observation.events:
        lines.append(f"recent_events: {' | '.join(observation.events[-3:])}")

    if observation.metadata.get("reward_breakdown"):
        breakdown = observation.metadata["reward_breakdown"]
        lines.append(
            "last_reward_breakdown: "
            + ", ".join(f"{key}={value}" for key, value in breakdown.items())
        )

    lines.append("patients:")
    for patient in observation.patients[:max_patients]:
        vitals = patient.vitals
        lines.append(
            "  - "
            f"{patient.patient_id}: complaint={patient.chief_complaint}, age={patient.age}, "
            f"esi={patient.assigned_esi}, loc={patient.location}, wait={patient.wait_time_min}, "
            f"news2={patient.news2_score}, qsofa={patient.qsofa_score}, "
            f"hr={vitals.hr}, sbp={vitals.sbp}, rr={vitals.rr}, o2={vitals.o2_sat}, temp={vitals.temp_c}, "
            f"tests={sorted(patient.completed_tests.keys())}"
        )

    if len(observation.patients) > max_patients:
        lines.append(f"  - ... {len(observation.patients) - max_patients} more patients omitted")

    return "\n".join(lines)


def make_user_prompt(
    dataset_prompt: str,
    observation: ERObservation,
    step_num: int,
    max_episode_steps: int,
    max_patients: int,
) -> str:
    """Build the user prompt for a single OpenER rollout step."""

    observation_text = render_observation(observation, max_patients=max_patients)
    return (
        f"Episode objective: {dataset_prompt}\n\n"
        f"Current turn: {step_num + 1} / {max_episode_steps}\n"
        "Decide the next 5-minute ER action bundle.\n\n"
        f"{observation_text}\n\n"
        "Return JSON only."
    )


def build_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    """Apply the tokenizer chat template when available."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return f"system: {system_prompt}\n\nuser: {user_prompt}\n\nassistant:"


def parse_action_json(text: str) -> ERAction:
    """Parse model text into an `ERAction`, falling back to a no-op action."""

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = brace.group(0) if brace else ""

    if not candidate:
        return ERAction(commands=[])

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict) and "action" in payload and isinstance(payload["action"], dict):
            payload = payload["action"]
        return ERAction(**payload)
    except Exception:
        return ERAction(commands=[])


def compute_combined_reward(
    dense_return: float,
    benchmark_score: float,
    config: OpenERTRLConfig,
) -> float:
    """Combine dense reward and terminal benchmark score into one bounded signal."""

    dense_component = math.tanh(dense_return / config.dense_reward_scale)
    return config.benchmark_weight * benchmark_score + config.dense_reward_weight * dense_component


def _decode_completion(output: dict[str, Any], tokenizer: Any) -> str:
    if output.get("text"):
        return str(output["text"])
    return tokenizer.decode(output["completion_ids"], skip_special_tokens=True)


def rollout_once(
    trainer: Any,
    env: Any,
    tokenizer: Any,
    dataset_prompt: str,
    config: OpenERTRLConfig,
    seed: int,
) -> dict[str, Any]:
    """Run one OpenER episode and return rollout artifacts for GRPO."""

    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset(task_id=config.task_id, seed=seed)

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    parsed_actions: list[dict[str, Any]] = []
    completion_texts: list[str] = []

    for step_num in range(config.max_episode_steps):
        if result.done:
            break

        user_prompt = make_user_prompt(
            dataset_prompt=dataset_prompt,
            observation=result.observation,
            step_num=step_num,
            max_episode_steps=config.max_episode_steps,
            max_patients=config.max_patients_in_prompt,
        )
        prompt_text = build_chat_prompt(tokenizer, config.system_prompt, user_prompt)

        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])

        completion_text = _decode_completion(rollout_output, tokenizer)
        action = parse_action_json(completion_text)

        parsed_actions.append(action.model_dump())
        completion_texts.append(completion_text)

        result = env.step(action)
        step_rewards.append(float(result.reward or 0.0))

    state = env.state()
    benchmark_score = float(state.benchmark_score or 0.0)
    dense_return = float(sum(step_rewards))
    combined_reward = compute_combined_reward(
        dense_return=dense_return,
        benchmark_score=benchmark_score,
        config=config,
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "step_rewards": step_rewards,
        "dense_return": dense_return,
        "benchmark_score": benchmark_score,
        "combined_reward": combined_reward,
        "parsed_actions": parsed_actions,
        "completion_texts": completion_texts,
    }


def build_grpo_rollout(config: OpenERTRLConfig | None = None):
    """Create TRL-compatible rollout and default reward functions for OpenER."""

    cfg = config or OpenERTRLConfig()
    seed_counter = {"value": cfg.seed_start}

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, list]:
        tokenizer = trainer.processing_class

        prompt_ids_batches: list[list[int]] = []
        completion_ids_batches: list[list[int]] = []
        logprobs_batches: list[list[float]] = []
        dense_returns: list[float] = []
        benchmark_scores: list[float] = []
        combined_rewards: list[float] = []
        parsed_actions: list[list[dict[str, Any]]] = []
        step_reward_traces: list[list[float]] = []
        completion_texts: list[list[str]] = []

        with ExitStack() as stack:
            for prompt in prompts:
                env = stack.enter_context(OpenEREnv(base_url=cfg.base_url).sync())
                seed = seed_counter["value"]
                seed_counter["value"] += 1

                episode = rollout_once(
                    trainer=trainer,
                    env=env,
                    tokenizer=tokenizer,
                    dataset_prompt=prompt_to_text(prompt) or cfg.dataset_prompt,
                    config=cfg,
                    seed=seed,
                )

                prompt_ids_batches.append(episode["prompt_ids"])
                completion_ids_batches.append(episode["completion_ids"])
                logprobs_batches.append(episode["logprobs"])
                dense_returns.append(episode["dense_return"])
                benchmark_scores.append(episode["benchmark_score"])
                combined_rewards.append(episode["combined_reward"])
                parsed_actions.append(episode["parsed_actions"])
                step_reward_traces.append(episode["step_rewards"])
                completion_texts.append(episode["completion_texts"])

        return {
            "prompt_ids": prompt_ids_batches,
            "completion_ids": completion_ids_batches,
            "logprobs": logprobs_batches,
            "dense_return": dense_returns,
            "benchmark_score": benchmark_scores,
            "combined_reward": combined_rewards,
            "parsed_actions": parsed_actions,
            "step_rewards": step_reward_traces,
            "completion_texts": completion_texts,
        }

    return rollout_func, reward_combined


def reward_combined(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("combined_reward", [])
    if rewards:
        return [float(reward) for reward in rewards]
    return [0.0] * len(completions)


def reward_benchmark(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("benchmark_score", [])
    if rewards:
        return [float(reward) for reward in rewards]
    return [0.0] * len(completions)


def reward_dense_return(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = kwargs.get("dense_return", [])
    if rewards:
        return [float(reward) for reward in rewards]
    return [0.0] * len(completions)


def build_dataset(
    dataset_prompt: str = DEFAULT_DATASET_PROMPT,
    dataset_size: int = 256,
):
    """Create a simple repeated-prompt training dataset for GRPO."""

    from datasets import Dataset

    return Dataset.from_dict({"prompt": [dataset_prompt] * dataset_size})


def build_grpo_config(
    output_dir: str = "open-er-grpo",
    **overrides: Any,
):
    """Build a sensible default `GRPOConfig` for OpenER training in Colab."""

    from trl import GRPOConfig

    defaults = {
        "output_dir": output_dir,
        "max_steps": 100,
        "learning_rate": 5e-6,
        "warmup_steps": 10,
        "per_device_train_batch_size": 1,
        "num_generations": 4,
        "generation_batch_size": 4,
        "max_completion_length": 256,
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.2,
        "logging_steps": 1,
        "report_to": "none",
    }
    defaults.update(overrides)
    return GRPOConfig(**defaults)


def create_grpo_trainer(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset: Any | None = None,
    dataset_prompt: str = DEFAULT_DATASET_PROMPT,
    dataset_size: int = 256,
    env_config: OpenERTRLConfig | None = None,
    grpo_config: Any | None = None,
    reward_funcs: list[Callable[..., list[float]]] | None = None,
):
    """Create a ready-to-train `GRPOTrainer` for OpenER."""

    from trl import GRPOTrainer

    cfg = env_config or OpenERTRLConfig(dataset_prompt=dataset_prompt)
    train_dataset = dataset or build_dataset(dataset_prompt=dataset_prompt, dataset_size=dataset_size)
    trainer_config = grpo_config or build_grpo_config()
    rollout_func, default_reward = build_grpo_rollout(cfg)

    return GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs or [default_reward],
        train_dataset=train_dataset,
        args=trainer_config,
        rollout_func=rollout_func,
    )


def train_open_er_grpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_prompt: str = DEFAULT_DATASET_PROMPT,
    dataset_size: int = 256,
    env_config: OpenERTRLConfig | None = None,
    grpo_config: Any | None = None,
    reward_funcs: list[Callable[..., list[float]]] | None = None,
):
    """Create a trainer and immediately start GRPO training."""

    trainer = create_grpo_trainer(
        model_name=model_name,
        dataset_prompt=dataset_prompt,
        dataset_size=dataset_size,
        env_config=env_config,
        grpo_config=grpo_config,
        reward_funcs=reward_funcs,
    )
    stats = trainer.train()
    return trainer, stats
