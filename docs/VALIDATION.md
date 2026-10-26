# Validation scope

The public repository includes portable regression tests for the DSP/model policies and static build checks in the macOS build script.

## Portable tests

Run:

```bash
bash Tests/portable/run_all.sh
```

Coverage includes:

- exact compact-bin versus full-bin learned forward equivalence;
- learned total weight-sum bounds `[0.5, 2.0]`;
- paper-faithful local Delaunay/DirAC interpolation;
- dynamic node counts 1–5;
- `4N` multi-node audio ingress;
- no Eigen allocation in the scheduled frame path test configuration;
- adaptive scheduler and preload policies;
- temporal re-entry and blend safety;
- query walking dynamics and continuity policies;
- simulated Apple/vDSP branch.

## macOS build checks

`BUILD_UNIVERSAL.command` checks the generated Xcode project before compilation and validates produced binaries for:

- `x86_64` and `arm64` slices;
- macOS 10.13 deployment metadata;
- absence of `cblas_*` and `$NEWLAPACK` dependencies in the learned path;
- expected `vDSP_mmul` backend marker.

Portable/static tests do not replace real DAW validation. Host-specific loading, routing and real-time performance should be tested on the target macOS systems and DAWs used for deployment.
