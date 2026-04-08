---
title: OpenER
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /docs
tags:
  - openenv
  - healthcare
  - triage
---

# OpenER

OpenER is an OpenEnv benchmark for emergency department triage and resource allocation. The agent manages a simulated ER shift with incomplete information, constrained beds and diagnostics, and patients who may deteriorate if they are mishandled.

## Why This Exists

This benchmark targets a real operational decision problem rather than a game. The agent has to:

- identify which patients are likely urgent from visible vitals
- allocate scarce beds and diagnostics
- order the right tests without wasting resources
- choose when to admit, discharge, or escalate

The environment exposes three official tasks:

- `easy_single_critical`
- `medium_evening_rush`
- `hard_capacity_crunch`

Each task returns dense step rewards for partial progress and a normalized terminal benchmark score in `0.0-1.0`.

## Reward Design

OpenER uses a two-layer scoring design:

- **Dense step reward** for learning and policy iteration
- **Terminal benchmark score** for official task evaluation

The dense reward is not based on a single clinical score. It combines:

- **guideline anchors**: ESI-style urgency expectations plus NEWS2 and qSOFA severity cues
- **workflow anchors**: wait-time expectations inspired by public ED operations benchmarks
- **hidden-truth anchors**: the simulator's concealed diagnosis, trajectory, and correct disposition

Dense reward components:

- `patient_safety_delta`
- `timeliness_delta`
- `diagnostic_quality_delta`
- `resource_efficiency_delta`
- `disposition_quality_delta`

Safety is intentionally asymmetric:

- undertriage is penalized more than overtriage
- unsafe discharge is penalized more than unnecessary testing
- missed visible deterioration is penalized more than low-acuity throughput misses

The terminal benchmark grader reports:

- `patient_safety`
- `timeliness`
- `diagnostic_quality`
- `resource_stewardship`

as a normalized score in `0.0-1.0`.

## Environment API

### Action Space

The agent submits an `ERAction` containing zero or more per-patient commands.

| Field | Type | Meaning |
|---|---|---|
| `patient_id` | `str` | Which patient to act on |
| `assign_bed` | `bool` | Move a waiting patient into a bed if capacity exists |
| `new_esi` | `int \| None` | Update the visible triage level to 1-5 |
| `order_tests` | `list[str]` | Order labs or imaging |
| `disposition` | `"home" \| "admit" \| None` | Discharge or admit |
| `call_specialist` | `str \| None` | Request specialty input |

### Observation Space

Each `ERObservation` includes:

- `task_id`
- `difficulty`
- `shift_minute`
- visible patients
- visible resources
- alerts
- event log
- inherited `reward`, `done`, and `metadata`

Observation metadata also includes a `reward_breakdown` dictionary so reward shaping remains inspectable during debugging and evaluation.

Each patient includes:

- complaint and demographics
- wait time
- visible vitals
- visible ESI
- completed tests
- NEWS2 and qSOFA summaries

### State Metadata

`state()` returns aggregate episode metadata only. It does not expose hidden diagnoses.

## Local Setup

```bash
uv sync
uv run server
```

Then open:

- `http://localhost:7860/health`
- `http://localhost:7860/docs`

The Hugging Face Space opens at `/docs`.

## Baseline Evaluation

Run the deterministic heuristic baseline:

```bash
uv run python -m scripts.eval_baselines --policy heuristic --episodes 10
```

This evaluates all three official tasks across fixed seeds and writes a JSON report under `outputs/evals/`.

Current 10-seed benchmark result:

| Task | Heuristic score | Random score |
|---|---:|---:|
| `easy_single_critical` | `0.590` | `0.394` |
| `medium_evening_rush` | `0.557` | `0.431` |
| `hard_capacity_crunch` | `0.501` | `0.428` |
| `macro_average` | `0.549` | `0.418` |

Seed set:

```text
[11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
```

## Example

```python
from open_er import ERAction, OpenEREnv

with OpenEREnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_id="easy_single_critical", seed=11)
    result = env.step(ERAction(commands=[]))
    print(result.observation.task_id)
    print(result.reward)
```

## Deployment

Validate locally:

```bash
openenv validate --verbose
```

Push to Hugging Face Spaces:

```bash
openenv push --repo-id <your-org-or-user>/open-er
```

Notes:

- local Docker verification requires `docker` to be installed on the host machine
- `openenv push` requires a logged-in Hugging Face CLI session or token

## Non-Goals

- custom dashboard
- raw clinical datasets
- LLM-based grading
- GRPO training pipeline

## Calibration Notes

The runtime environment is deterministic and self-contained. External datasets and standards are used as **design anchors**, not live dependencies:

- ESI-style triage expectations for urgency and resource use
- NEWS2 and qSOFA for visible deterioration signals
- public ED wait-time/crowding literature for timeliness calibration
- MIMIC-IV-ED / MIETIC / NHAMCS as offline calibration references only
