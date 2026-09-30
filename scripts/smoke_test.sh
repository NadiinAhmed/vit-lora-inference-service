#!/usr/bin/env bash
# Smoke test a RUNNING inference service over HTTP, like a real client would.
# Used twice by the pipeline: on the freshly built image (CI) and on the
# deployed image pulled from the registry (CD verification).
#
# Usage: scripts/smoke_test.sh <base_url> <image_path> [expected_label]
set -euo pipefail

BASE_URL="${1:?usage: smoke_test.sh <base_url> <image_path> [expected_label]}"
IMAGE="${2:?image path required}"
EXPECTED_LABEL="${3:-}"
TMP="$(mktemp -d)"

# 1) Wait until the service is up. The model loads at startup, so this can take a few seconds.
echo "Waiting for ${BASE_URL}/health ..."
for attempt in $(seq 1 60); do
  if curl -fsS "${BASE_URL}/health" -o "${TMP}/health.json" 2>/dev/null; then
    break
  fi
  if [ "${attempt}" -eq 60 ]; then
    echo "FAIL  service did not answer /health within 120 s"
    exit 1
  fi
  sleep 2
done
echo "health:  $(cat "${TMP}/health.json")"

# 2) /health must report a loaded model with the 6 classes it was trained on.
python3 - "${TMP}/health.json" <<'PY'
import json, sys
health = json.load(open(sys.argv[1]))
assert health["status"] == "ok", "status is not ok"
assert health["model_loaded"] is True, "model is not loaded"
assert len(health["classes"]) == 6, f"expected 6 classes, got {health['classes']}"
print("PASS  /health: model loaded, 6 classes")
PY

# 3) /predict with a real image must return a valid, well-formed prediction.
curl -fsS -X POST "${BASE_URL}/predict" -F "file=@${IMAGE};type=image/jpeg" -o "${TMP}/predict.json"
echo "predict: $(cat "${TMP}/predict.json")"

python3 - "${TMP}/health.json" "${TMP}/predict.json" "${EXPECTED_LABEL}" <<'PY'
import json, sys
classes = json.load(open(sys.argv[1]))["classes"]
prediction = json.load(open(sys.argv[2]))
expected = sys.argv[3]
assert prediction["label"] in classes, f"unknown label {prediction['label']}"
assert 0.0 <= prediction["confidence"] <= 1.0, "confidence outside [0, 1]"
assert prediction["label"] == prediction["top_k"][0]["label"], "label is not the top-1 class"
print(f"PASS  /predict: label={prediction['label']} confidence={prediction['confidence']}")
if expected:
    assert prediction["label"] == expected, f"expected '{expected}', got '{prediction['label']}'"
    print(f"PASS  /predict: sample image correctly classified as '{expected}'")
PY

# 4) Input validation still works in the running container: non-images are rejected.
echo "not an image" > "${TMP}/notes.txt"
status="$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/predict" -F "file=@${TMP}/notes.txt;type=text/plain")"
if [ "${status}" != "415" ]; then
  echo "FAIL  non-image upload returned ${status}, expected 415"
  exit 1
fi
echo "PASS  /predict rejects non-images (415)"

echo "Smoke test passed."
