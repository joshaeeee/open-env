from open_er.models import ERAction, PatientCommand
from open_er.server.environment import ERTriageEnvironment


def _first_patient(env: ERTriageEnvironment, predicate):
    return next(patient for patient in env._patients.values() if predicate(patient))


def _neutralize_background_risk(env: ERTriageEnvironment, focus_patient_id: str) -> None:
    env._future_arrivals = []
    for patient in env._patients.values():
        if patient.patient_id == focus_patient_id:
            continue
        patient.true_esi = max(patient.true_esi, 4)
        patient.assigned_esi = max(patient.assigned_esi, 4)
        patient.trajectory = "stable"


def test_unsafe_discharge_is_worse_than_unnecessary_test():
    env_bad = ERTriageEnvironment()
    env_bad.reset(task_id="easy_single_critical", seed=11)
    critical = _first_patient(env_bad, lambda patient: patient.true_esi <= 2)
    _neutralize_background_risk(env_bad, critical.patient_id)
    obs_bad = env_bad.step(
        ERAction(commands=[PatientCommand(patient_id=critical.patient_id, disposition="home")])
    )

    env_minor = ERTriageEnvironment()
    env_minor.reset(task_id="easy_single_critical", seed=11)
    low_acuity = _first_patient(env_minor, lambda patient: patient.true_esi >= 4)
    _neutralize_background_risk(env_minor, low_acuity.patient_id)
    obs_minor = env_minor.step(
        ERAction(commands=[PatientCommand(patient_id=low_acuity.patient_id, order_tests=["ct_head"])])
    )

    assert float(obs_bad.reward or 0.0) < float(obs_minor.reward or 0.0)
    assert abs(float(obs_bad.reward or 0.0)) > abs(float(obs_minor.reward or 0.0))


def test_undertriage_penalty_is_worse_than_overtriage_penalty():
    env_under = ERTriageEnvironment()
    env_under.reset(task_id="easy_single_critical", seed=11)
    critical = _first_patient(env_under, lambda patient: patient.true_esi <= 2)
    _neutralize_background_risk(env_under, critical.patient_id)
    worse_esi = min(5, critical.assigned_esi + 1)
    obs_under = env_under.step(
        ERAction(commands=[PatientCommand(patient_id=critical.patient_id, new_esi=worse_esi)])
    )

    env_over = ERTriageEnvironment()
    env_over.reset(task_id="easy_single_critical", seed=11)
    stable = _first_patient(env_over, lambda patient: patient.true_esi >= 3)
    _neutralize_background_risk(env_over, stable.patient_id)
    more_urgent = max(1, stable.assigned_esi - 1)
    obs_over = env_over.step(
        ERAction(commands=[PatientCommand(patient_id=stable.patient_id, new_esi=more_urgent)])
    )

    assert abs(float(obs_under.reward or 0.0)) > abs(float(obs_over.reward or 0.0))


def test_reward_breakdown_metadata_present():
    env = ERTriageEnvironment()
    env.reset(task_id="easy_single_critical", seed=11)
    obs = env.step(ERAction(commands=[]))
    breakdown = obs.metadata["reward_breakdown"]
    assert set(breakdown) == {
        "patient_safety_delta",
        "timeliness_delta",
        "diagnostic_quality_delta",
        "resource_efficiency_delta",
        "disposition_quality_delta",
    }
