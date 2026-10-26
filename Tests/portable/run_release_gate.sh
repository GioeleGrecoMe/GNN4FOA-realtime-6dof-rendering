#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
CXX="${CXX:-g++}"
BASE=(-O2 -std=c++17 -I"$PWD/fake" -I"$ROOT/Source" -I"$ROOT/Source/Eigen")
ENG="$ROOT/Source/DiracGNNEngine.cpp"
OUT="${TMPDIR:-/tmp}/gnn4foa_v2_release_gate"
rm -rf "$OUT" && mkdir -p "$OUT"

HOST_LINK=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  HOST_LINK=(-framework Accelerate)
  echo "Host backend: Apple Accelerate/vDSP (linking -framework Accelerate)"
else
  echo "Host backend: portable scalar GEMM"
fi

echo "== source structure =="
python3 "$PWD/check_cpp_structure.py"

echo "== legacy Intel/macOS policy =="
python3 "$PWD/check_legacy_macos_policy.py" "$ROOT"

echo "== host backend compile/link/run =="
"$CXX" "${BASE[@]}" -c "$ENG" -o "$OUT/engine_host.o"
"$CXX" "${BASE[@]}" test_backend_smoke.cpp "$OUT/engine_host.o" "${HOST_LINK[@]}" -o "$OUT/backend_host"
"$OUT/backend_host"

echo "== learned weight-sum policy =="
"$CXX" "${BASE[@]}" test_weight_sum_bounds.cpp "$OUT/engine_host.o" "${HOST_LINK[@]}" -o "$OUT/weight_sum"
"$OUT/weight_sum"

echo "== manual FOA topology policy =="
"$CXX" -O2 -std=c++17 -I"$ROOT/Source" test_manual_foa_policy.cpp -o "$OUT/manual_foa"
"$OUT/manual_foa"

echo "== audio-time scheduler policy =="
"$CXX" -O2 -std=c++17 test_audio_time_scheduler.cpp -o "$OUT/audio_scheduler"
"$OUT/audio_scheduler"

echo "== forced portable fallback compile/link/run =="
"$CXX" "${BASE[@]}" -DGNN4FOA_FORCE_PORTABLE_GEMM=1 -c "$ENG" -o "$OUT/engine_portable.o"
"$CXX" "${BASE[@]}" -DGNN4FOA_FORCE_PORTABLE_GEMM=1 test_backend_smoke.cpp "$OUT/engine_portable.o" -o "$OUT/backend_portable"
"$OUT/backend_portable"

echo "== simulated Apple/vDSP compile/link/run =="
APPLE=(-O2 -D__APPLE__ -std=c++17 -I"$PWD/fake_apple" -I"$PWD/fake" -I"$ROOT/Source" -I"$ROOT/Source/Eigen")
"$CXX" "${APPLE[@]}" -c "$ENG" -o "$OUT/engine_apple.o"
"$CXX" "${APPLE[@]}" test_backend_smoke.cpp "$OUT/engine_apple.o" -o "$OUT/backend_apple"
"$OUT/backend_apple"

echo "PASS: fail-fast release gate completed"
