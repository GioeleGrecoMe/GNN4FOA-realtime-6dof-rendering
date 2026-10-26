#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
CXX="${CXX:-g++}"
BASE=(-O3 -std=c++17 -I"$PWD/fake" -I"$ROOT/Source" -I"$ROOT/Source/Eigen")
ENG="$ROOT/Source/DiracGNNEngine.cpp"
OUT="${TMPDIR:-/tmp}/gnn4foa_v2_extended"
rm -rf "$OUT" && mkdir -p "$OUT"

HOST_LINK=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  HOST_LINK=(-framework Accelerate)
fi

# Correctness and release-safety gates first.
bash "$PWD/run_all.sh"

"$CXX" "${BASE[@]}" -c "$ENG" -o "$OUT/engine.o"
link_run () {
  local src="$1"; shift
  local exe="$OUT/$(basename "$src" .cpp)"
  "$CXX" "${BASE[@]}" "$src" "$OUT/engine.o" "${HOST_LINK[@]}" -o "$exe"
  "$exe" "$@"
}

echo "== FULL multirate tracking sweep =="
link_run test_multirate_quality.cpp

echo "== FULL forward scaling benchmark =="
link_run bench_forward.cpp

echo "== threshold classification benchmark =="
"$CXX" -O3 -DNDEBUG -std=c++17 bench_threshold.cpp -o "$OUT/bench_threshold"
"$OUT/bench_threshold"

echo "PASS: extended portable validation completed"
