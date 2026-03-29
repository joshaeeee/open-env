from open_er.models import ERAction, ERObservation, ERState, PatientCommand, ResourceView


def test_action_round_trip():
    action = ERAction(commands=[PatientCommand(patient_id="pt_1", assign_bed=True, order_tests=["ecg"])])
    payload = action.model_dump()
    restored = ERAction(**payload)
    assert restored.commands[0].assign_bed is True
    assert restored.commands[0].order_tests == ["ecg"]


def test_observation_defaults():
    observation = ERObservation(
        task_id="easy_single_critical",
        difficulty="easy",
        shift_minute=0,
        patients=[],
        resources=ResourceView(
            beds_available=1,
            beds_total=1,
            ct_available_in_min=0,
            lab_queue_length=0,
            specialist_available={},
        ),
    )
    assert observation.done is False
    assert observation.reward is None


def test_state_safe_defaults():
    state = ERState()
    assert state.benchmark_breakdown == {}
