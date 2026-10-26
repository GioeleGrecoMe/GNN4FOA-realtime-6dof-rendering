# Validation scope

The public repository includes portable regression tests for the DSP/model policies and static build checks in the macOS build script.

## Portable tests

Run:

```bash
bash Tests/portable/run_all.sh
```

Coverage includes:

- first-party C/C++ delimiter integrity for `Source/` (excluding vendored Eigen), catching unmatched braces/parentheses/brackets before Xcode;
- exact compact-bin versus full-bin learned forward equivalence;
- learned total weight-sum bounds `[0.5, 2.0]`;
- paper-faithful local Delaunay/DirAC interpolation;
- dynamic node counts 1–10;
- `4N` multi-node audio ingress;
- no Eigen allocation in the scheduled frame path test configuration;
- adaptive scheduler and preload policies;
- temporal re-entry and learned-weight blend safety;
- 3 dB GNN/Delaunay gate hysteresis and bidirectional 16 ms transition;
- missing-bin-only coverage refresh policy and bin-count- and node-count-aware admission;
- query walking dynamics and continuity policies;
- simulated Apple/vDSP branch;
- forced portable GEMM fallback even on Apple-oriented builds;
- legacy Intel/macOS policy guard: Universal `x86_64 + arm64`, macOS 10.13 floor, Accelerate linkage, and no explicit native/AVX/FMA compiler tuning.

## macOS build checks

`BUILD_UNIVERSAL.command` first runs `Tests/portable/run_release_gate.sh`, then checks the generated Xcode project and validates produced binaries for:

- `x86_64` and `arm64` slices;
- macOS 10.13 deployment metadata;
- conservative Intel vector settings appropriate to 2010-era x86_64 CPUs;
- absence of AVX/VEX instruction mnemonics in the produced x86_64 slice;
- absence of `cblas_*` and `$NEWLAPACK` dependencies in the learned path;
- expected `vDSP_mmul` backend marker.

Portable/static tests do not replace real DAW validation. Host-specific loading, routing and real-time performance should be tested on the target macOS systems and DAWs used for deployment.

The GitHub Actions workflow `.github/workflows/macos-build.yml` also runs the complete `BUILD_UNIVERSAL.command` on a macOS runner and uploads the installer only after the Universal AU/VST3 build and binary checks succeed.

## v2.0.0 release evidence

- [`validation/RELEASE_VALIDATION_v2.0.0.txt`](validation/RELEASE_VALIDATION_v2.0.0.txt) — consolidated release checks and limitations.
- [`validation/PORTABLE_FORWARD_BENCH_v2.0.0.txt`](validation/PORTABLE_FORWARD_BENCH_v2.0.0.txt) — generic portable-backend scaling reference; not a Mac/vDSP benchmark.
- [`validation/FAKE_APPLE_VDSP_v2.0.0.txt`](validation/FAKE_APPLE_VDSP_v2.0.0.txt) — simulated Apple/vDSP branch for N=2..10.
- [`validation/LEGACY_FORWARD_EQUIVALENCE_v2.0.0.txt`](validation/LEGACY_FORWARD_EQUIVALENCE_v2.0.0.txt) — exact v1.0.1↔v2 learned-forward equivalence for N=2..5.

## Test tiers

- `bash Tests/portable/run_release_gate.sh`: fast, blocking source/backend/legacy compatibility gate used before Xcode.
- `bash Tests/portable/run_all.sh`: broader portable correctness and policy suite.
- `bash Tests/portable/run_extended.sh`: full multirate and forward-scaling performance sweeps in addition to the correctness suite.

The portable scalar backend is intentionally much slower than Accelerate/vDSP; long benchmark duration on non-Apple CI is therefore not interpreted as a DSP correctness failure.
