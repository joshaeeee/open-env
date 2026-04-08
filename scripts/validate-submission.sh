#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Accepts any of:
#   - Hugging Face Space page URL: https://huggingface.co/spaces/<owner>/<space>
#   - Hugging Face runtime URL:    https://<owner>-<space>.hf.space
#   - Repo slug:                   <owner>/<space>
#
# Run:
#   ./scripts/validate-submission.sh <space_url_or_slug> [repo_dir]
#
# Examples:
#   ./scripts/validate-submission.sh https://huggingface.co/spaces/joshaeeee/open-er
#   ./scripts/validate-submission.sh https://joshaeeee-open-er.hf.space
#   ./scripts/validate-submission.sh joshaeeee/open-er .
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
INFERENCE_TIMEOUT_PER_TASK=300

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$secs" "$@" <<'PY'
import subprocess
import sys

timeout_s = float(sys.argv[1])
command = sys.argv[2:]

try:
    completed = subprocess.run(command, timeout=timeout_s, check=False)
    raise SystemExit(completed.returncode)
except subprocess.TimeoutExpired:
    raise SystemExit(124)
PY
  else
    "$@" &
    local pid=$!
    (
      sleep "$secs"
      kill "$pid" 2>/dev/null
    ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return "$rc"
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() {
  rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"
}
trap cleanup EXIT

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

resolve_ping_url() {
  local raw="$1"
  raw="${raw%/}"

  if [[ "$raw" =~ ^https://[^/]+\.hf\.space$ ]]; then
    printf "%s" "$raw"
    return 0
  fi

  if [[ "$raw" =~ ^https://huggingface\.co/spaces/([^/]+)/([^/]+)$ ]]; then
    printf "https://%s-%s.hf.space" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi

  if [[ "$raw" =~ ^([^/]+)/([^/]+)$ ]]; then
    printf "https://%s-%s.hf.space" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi

  return 1
}

find_openenv_cmd() {
  if command -v openenv >/dev/null 2>&1; then
    printf "openenv"
    return 0
  fi

  if command -v uv >/dev/null 2>&1 && [ -f "$REPO_DIR/pyproject.toml" ]; then
    printf "uv run --project %q openenv" "$REPO_DIR"
    return 0
  fi

  return 1
}

setup_project_python_cmd() {
  if command -v uv >/dev/null 2>&1 && [ -f "$REPO_DIR/pyproject.toml" ]; then
    PROJECT_PYTHON_CMD=(uv run --project "$REPO_DIR" python)
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    PROJECT_PYTHON_CMD=(python3)
    return 0
  fi

  return 1
}

SPACE_INPUT="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$SPACE_INPUT" ]; then
  printf "Usage: %s <space_url_or_slug> [repo_dir]\n" "$0"
  printf "\n"
  printf "  space_url_or_slug   HF Space page URL, .hf.space URL, or owner/space\n"
  printf "  repo_dir            Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

if ! PING_URL="$(resolve_ping_url "$SPACE_INPUT")"; then
  printf "Error: could not interpret Space input '%s'\n" "$SPACE_INPUT"
  printf "Expected a HF page URL, a .hf.space URL, or owner/space.\n"
  exit 1
fi

export PING_URL
PASS=0
PROJECT_PYTHON_CMD=()

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:        $REPO_DIR"
log "Space input: $SPACE_INPUT"
log "Ping URL:    $PING_URL"
printf "\n"

log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_STDOUT=$(portable_mktemp "validate-curl-out")
CURL_STDERR=$(portable_mktemp "validate-curl-err")
CLEANUP_FILES+=("$CURL_STDOUT" "$CURL_STDERR")
HTTP_CODE=$(curl -sS -o "$CURL_STDOUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_STDERR" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  if [ -s "$CURL_STDOUT" ]; then
    printf "%s\n" "$(cat "$CURL_STDOUT")"
  fi
  hint "Make sure your Space is running and the URL is correct."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/5: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if ! docker info >/dev/null 2>&1; then
  fail "docker daemon is not running"
  hint "Start Docker Desktop or your Docker daemon, then rerun validation."
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/5: Running openenv validate${NC} ..."

if ! OPENENV_CMD="$(find_openenv_cmd)"; then
  fail "openenv command not found"
  hint "Install it globally with: pip install openenv-core"
  hint "Or install uv and keep a project-local pyproject.toml so the validator can use 'uv run openenv'."
  stop_at "Step 3"
fi

log "  Using validator command: $OPENENV_CMD validate"

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && eval "$OPENENV_CMD validate" 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  if [ -n "$VALIDATE_OUTPUT" ]; then
    log "  $VALIDATE_OUTPUT"
  fi
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/5: Checking inference.py contract${NC} ..."

if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "Missing root-level inference.py"
  hint "Place the submitted inference runner at $REPO_DIR/inference.py"
  stop_at "Step 4"
fi

if ! grep -q 'API_BASE_URL' "$REPO_DIR/inference.py"; then
  fail "inference.py does not reference API_BASE_URL"
  stop_at "Step 4"
fi

if ! grep -q 'MODEL_NAME' "$REPO_DIR/inference.py"; then
  fail "inference.py does not reference MODEL_NAME"
  stop_at "Step 4"
fi

if ! grep -q 'HF_TOKEN' "$REPO_DIR/inference.py"; then
  fail "inference.py does not reference HF_TOKEN"
  stop_at "Step 4"
fi

if ! grep -Eq 'from openai import OpenAI|import openai' "$REPO_DIR/inference.py"; then
  fail "inference.py does not use the OpenAI client"
  stop_at "Step 4"
fi

if ! grep -q '\[START\]' "$REPO_DIR/inference.py" || ! grep -q '\[STEP\]' "$REPO_DIR/inference.py" || ! grep -q '\[END\]' "$REPO_DIR/inference.py"; then
  fail "inference.py is missing required structured log markers"
  stop_at "Step 4"
fi

if ! setup_project_python_cmd; then
  fail "Could not find a Python runtime for project-level checks"
  hint "Install uv or python3."
  stop_at "Step 4"
fi

if (cd "$REPO_DIR" && "${PROJECT_PYTHON_CMD[@]}" -m py_compile inference.py) >/dev/null 2>&1; then
  pass "inference.py exists, compiles, and references the required interface"
else
  fail "inference.py does not compile"
  stop_at "Step 4"
fi

log "${BOLD}Step 5/5: Running inference across official tasks${NC} ..."

TASKS_RAW=$(
  cd "$REPO_DIR" && "${PROJECT_PYTHON_CMD[@]}" - <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path.cwd()

try:
    from open_er.server.tasks import OFFICIAL_TASKS
except Exception:
    spec = importlib.util.spec_from_file_location(
        "open_er",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_er"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from open_er.server.tasks import OFFICIAL_TASKS

for task_name in OFFICIAL_TASKS:
    print(task_name)
PY
)

TASKS=$(printf "%s\n" "$TASKS_RAW" | sed '/^[[:space:]]*$/d')
TASK_COUNT=$(printf "%s\n" "$TASKS" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')

if [ "$TASK_COUNT" -lt 3 ]; then
  fail "Expected at least 3 official tasks, found $TASK_COUNT"
  stop_at "Step 5"
fi

log "  Found $TASK_COUNT official tasks"

while IFS= read -r TASK_NAME; do
  [ -z "$TASK_NAME" ] && continue
  TASK_STDOUT=$(portable_mktemp "validate-inference-out")
  TASK_STDERR=$(portable_mktemp "validate-inference-err")
  CLEANUP_FILES+=("$TASK_STDOUT" "$TASK_STDERR")

  log "  Running inference.py for task=$TASK_NAME"

  if (
    cd "$REPO_DIR" && \
    run_with_timeout "$INFERENCE_TIMEOUT_PER_TASK" env \
      API_BASE_URL="http://127.0.0.1:1/v1" \
      MODEL_NAME="validator-smoke" \
      HF_TOKEN="dummy" \
      OPEN_ER_TASK="$TASK_NAME" \
      OPEN_ER_REQUEST_TIMEOUT_S="0.2" \
      OPEN_ER_MAX_RETRIES="0" \
      "${PROJECT_PYTHON_CMD[@]}" inference.py
  ) >"$TASK_STDOUT" 2>"$TASK_STDERR"; then
    :
  else
    fail "inference.py failed for task=$TASK_NAME"
    [ -s "$TASK_STDOUT" ] && printf "%s\n" "$(cat "$TASK_STDOUT")"
    [ -s "$TASK_STDERR" ] && printf "%s\n" "$(cat "$TASK_STDERR")"
    stop_at "Step 5"
  fi

  if ! python3 - "$TASK_STDOUT" "$TASK_NAME" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_name = sys.argv[2]
lines = path.read_text().splitlines()

if not lines:
    raise SystemExit("No stdout emitted")

start_re = re.compile(rf"^\[START\] task={re.escape(task_name)} env=open_er model=.+$")
step_re = re.compile(r"^\[STEP\] step=(\d+) action=.* reward=(-?\d+\.\d{2}) done=(true|false) error=null$")
end_re = re.compile(r"^\[END\] success=(true|false) steps=(\d+) score=(\d+\.\d{2}) rewards=(.*)$")

if not start_re.match(lines[0]):
    raise SystemExit(f"Invalid [START] line: {lines[0]}")

if len(lines) < 3:
    raise SystemExit("Expected at least one [STEP] line and one [END] line")

step_lines = lines[1:-1]
if not step_lines:
    raise SystemExit("Missing [STEP] lines")

for line in step_lines:
    if not step_re.match(line):
        raise SystemExit(f"Invalid [STEP] line: {line}")

end_match = end_re.match(lines[-1])
if not end_match:
    raise SystemExit(f"Invalid [END] line: {lines[-1]}")

success = end_match.group(1)
steps = int(end_match.group(2))
score = float(end_match.group(3))
rewards_field = end_match.group(4)
rewards = [] if rewards_field == "" else rewards_field.split(",")

if success != "true":
    raise SystemExit("Inference run did not report success=true")
if steps != len(step_lines):
    raise SystemExit(f"steps={steps} does not match emitted [STEP] lines={len(step_lines)}")
if not (0.0 <= score <= 1.0):
    raise SystemExit(f"Score out of range: {score}")
if len(rewards) != len(step_lines):
    raise SystemExit("Reward count does not match [STEP] lines")

print(f"score={score:.2f} steps={steps}")
PY
  then
    fail "Structured stdout validation failed for task=$TASK_NAME"
    [ -s "$TASK_STDOUT" ] && printf "%s\n" "$(cat "$TASK_STDOUT")"
    [ -s "$TASK_STDERR" ] && printf "%s\n" "$(cat "$TASK_STDERR")"
    stop_at "Step 5"
  fi
done <<< "$TASKS"

pass "inference.py completed successfully across all official tasks with valid structured logs"

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 5/5 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
