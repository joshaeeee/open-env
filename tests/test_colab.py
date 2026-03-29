from open_er.colab import (
    SYSTEM_PROMPT,
    OpenERTRLConfig,
    compute_combined_reward,
    make_user_prompt,
    parse_action_json,
    prompt_to_text,
    render_observation,
)
from open_er.server.environment import ERTriageEnvironment


def test_prompt_to_text_handles_chat_messages():
    prompt = [
        {"role": "system", "content": "You are a triage agent."},
        {"role": "user", "content": "Decide what to do next."},
    ]
    text = prompt_to_text(prompt)
    assert "system: You are a triage agent." in text
    assert "user: Decide what to do next." in text


def test_parse_action_json_extracts_payload():
    action = parse_action_json(
        """
        Here is my action:
        ```json
        {"commands": [{"patient_id": "pt_001_0", "assign_bed": true, "order_tests": ["ecg"]}]}
        ```
        """
    )
    assert len(action.commands) == 1
    assert action.commands[0].patient_id == "pt_001_0"
    assert action.commands[0].assign_bed is True
    assert action.commands[0].order_tests == ["ecg"]


def test_parse_action_json_falls_back_to_noop():
    action = parse_action_json("not valid json")
    assert action.commands == []


def test_render_observation_contains_key_fields():
    env = ERTriageEnvironment()
    observation = env.reset(task_id="easy_single_critical", seed=11)
    text = render_observation(observation, max_patients=3)
    assert "task_id: easy_single_critical" in text
    assert "patients:" in text
    assert "resources:" in text


def test_trl_config_defaults_are_stable():
    config = OpenERTRLConfig()
    assert config.task_id == "easy_single_critical"
    assert config.seed_start == 11


def test_system_prompt_mentions_json_only():
    assert "Output JSON only." in SYSTEM_PROMPT
    assert "VALID TEST NAMES" in SYSTEM_PROMPT
    assert "VALID SPECIALISTS" in SYSTEM_PROMPT


def test_make_user_prompt_includes_turn_and_objective():
    env = ERTriageEnvironment()
    observation = env.reset(task_id="easy_single_critical", seed=11)
    prompt = make_user_prompt(
        dataset_prompt="Keep critical patients safe.",
        observation=observation,
        step_num=1,
        max_episode_steps=8,
        max_patients=4,
    )
    assert "Episode objective: Keep critical patients safe." in prompt
    assert "Current turn: 2 / 8" in prompt
    assert "Return JSON only." in prompt


def test_compute_combined_reward_blends_dense_and_terminal():
    config = OpenERTRLConfig(dense_reward_scale=5.0, dense_reward_weight=0.5, benchmark_weight=1.0)
    reward = compute_combined_reward(dense_return=10.0, benchmark_score=0.6, config=config)
    assert reward > 0.6
