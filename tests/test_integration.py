import json
import subprocess
import sys
from pathlib import Path


def test_baseline_script_runs():
    output = subprocess.run(
        [sys.executable, "-m", "scripts.eval_baselines", "--policy", "heuristic", "--episodes", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "macro_average" in output.stdout
    path = Path("outputs/evals/heuristic_baseline.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["policy"] == "heuristic"
