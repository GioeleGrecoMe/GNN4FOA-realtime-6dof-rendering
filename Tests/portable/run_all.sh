#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
CXX="${CXX:-g++}"
BASE=(-O3 -std=c++17 -I"$PWD/fake" -I"$ROOT/Source" -I"$ROOT/Source/Eigen")
ENG="$ROOT/Source/DiracGNNEngine.cpp"
OUT="${TMPDIR:-/tmp}/gnn4foa_v2_tests"
rm -rf "$OUT" && mkdir -p "$OUT"

# On a real macOS host DiracGNNEngine.cpp selects Accelerate/vDSP. Compiling an
# object succeeds without framework flags, but every executable that links that
# object must also link Accelerate. Linux/Windows portable gates need no extra
# linker flags. Keeping this host-only prevents the previous false CI failure.
HOST_LINK=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  HOST_LINK=(-framework Accelerate)
  echo "Host backend: Apple Accelerate/vDSP (linking -framework Accelerate)"
else
  echo "Host backend: portable scalar GEMM"
fi

echo "== first-party C/C++ source structure =="
python3 "$PWD/check_cpp_structure.py"

echo "== legacy Intel/macOS build-policy guard =="
python3 "$PWD/check_legacy_macos_policy.py" "$ROOT"

echo "== compile engine once =="
"$CXX" "${BASE[@]}" -c "$ENG" -o "$OUT/engine.o"
link_run () {
  local src="$1"; shift
  local exe="$OUT/$(basename "$src" .cpp)"
  "$CXX" "${BASE[@]}" "$src" "$OUT/engine.o" "${HOST_LINK[@]}" -o "$exe"
  "$exe" "$@"
}

echo "== compact-bin exactness =="
link_run test_compact_equiv.cpp
echo "== learned weight-sum bounds [0.5, 2.0] =="
link_run test_weight_sum_bounds.cpp

echo "== paper-faithful Delaunay/DirAC baseline =="
link_run test_paper_delaunay.cpp

echo "== dynamic nodes =="
link_run test_dynamic_nodes.cpp
echo "== multi-node 4*N audio ingress =="
link_run test_multinode_audio_ingress.cpp
echo "== no Eigen allocation in scheduled frame path =="
link_run test_nomalloc_dynamic_process.cpp
echo "== scheduler decisions + overhead =="
link_run test_scheduler.cpp
echo "== multirate tracking quality (synthetic) =="
link_run test_multirate_quality.cpp --smoke
echo "== performance benchmarks =="
echo "SKIP in fail-fast release gate; run Tests/portable/run_extended.sh for full sweeps"

echo "== preload policy =="
"$CXX" -O3 -std=c++17 test_preload_policy.cpp -o "$OUT/test_preload_policy"
"$OUT/test_preload_policy"

echo "== manual FOA node-count policy =="
"$CXX" -O3 -std=c++17 -I"$ROOT/Source" test_manual_foa_policy.cpp -o "$OUT/test_manual_foa_policy"
"$OUT/test_manual_foa_policy"

echo "== temporal safety and fresh-bin policy =="
link_run test_temporal_safety.cpp

echo "== hybrid GNN/Delaunay gate + coverage safety =="
link_run test_hybrid_gate_safety.cpp
echo "== node-aware scheduler scaling to N=10 =="
link_run test_node_scaling_scheduler.cpp

echo "== audio-time scheduler policy =="
"$CXX" -O3 -std=c++17 test_audio_time_scheduler.cpp -o "$OUT/test_audio_time_scheduler"
"$OUT/test_audio_time_scheduler"

echo "== learned blend latency policy =="
"$CXX" -O3 -std=c++17 test_blend_latency_policy.cpp -o "$OUT/test_blend_latency_policy"
"$OUT/test_blend_latency_policy"

echo "== keyboard walking dynamics =="
"$CXX" -O3 -std=c++17 test_walk_motion.cpp -o "$OUT/test_walk_motion"
"$OUT/test_walk_motion"

echo "== continuity stress policy =="
"$CXX" -O3 -std=c++17 test_continuity_policy.cpp -o "$OUT/test_continuity_policy"
"$OUT/test_continuity_policy"

# Force and run the fallback independently even on a real Mac. This catches a
# future regression where the vDSP path works but the compatibility fallback
# stops compiling or diverges numerically.
echo "== forced portable GEMM fallback =="
"$CXX" "${BASE[@]}" -DGNN4FOA_FORCE_PORTABLE_GEMM=1 -c "$ENG" -o "$OUT/engine_portable.o"
"$CXX" "${BASE[@]}" -DGNN4FOA_FORCE_PORTABLE_GEMM=1 test_backend_smoke.cpp "$OUT/engine_portable.o" -o "$OUT/test_backend_smoke_portable"
"$OUT/test_backend_smoke_portable"

# Simulate the Apple compile-time branch on non-Apple CI without requiring the
# real framework. The fake header provides a reference vDSP_mmul implementation.
echo "== fake Apple/vDSP branch =="
APPLE=(-O3 -D__APPLE__ -std=c++17 -I"$PWD/fake_apple" -I"$PWD/fake" -I"$ROOT/Source" -I"$ROOT/Source/Eigen")
"$CXX" "${APPLE[@]}" -c "$ENG" -o "$OUT/engine_apple.o"
"$CXX" "${APPLE[@]}" test_backend_smoke.cpp "$OUT/engine_apple.o" -o "$OUT/test_backend_smoke_apple"
"$OUT/test_backend_smoke_apple"

echo "PASS: portable suite completed"
