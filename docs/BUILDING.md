# Building GNN4FOA on macOS

## Requirements

- full Xcode 12.4 or newer;
- Git, unless JUCE is supplied through `JUCE_ROOT`;
- enough disk space to clone JUCE 9.0.1 and build Universal AU/VST3 binaries.

The build host must be modern enough for JUCE 9/Xcode. A 2010-era Mac is treated as a **runtime target**, not as the machine on which this source release is compiled.

## One-command build

From the repository root:

```bash
bash BUILD_UNIVERSAL.command
```

The script:

1. validates Xcode and the generated Xcode project;
2. locates `ThirdParty/JUCE`, uses `JUCE_ROOT`, or clones JUCE tag `9.0.1`;
3. cleans only project-local build products;
4. runs a fail-fast release gate that compiles/links/runs the host matrix backend, the forced portable fallback and a simulated Apple/vDSP backend;
5. builds AU and VST3 for `x86_64 + arm64`;
6. requests macOS deployment target 10.13;
7. explicitly keeps Intel vector extensions at the conservative Xcode baseline and disables project-level SSE3/SSE4.x opt-ins;
8. checks architecture slices, deployment metadata and the x86_64 disassembly, rejecting AVX/VEX instructions that are unsafe on 2010-era Intel CPUs;
9. rejects CBLAS/NEWLAPACK-linked learned backends;
10. ad-hoc signs the local build;
11. creates `GNN4FOA_v2.0.0_macOS_Universal_installer.zip`.

To use a local JUCE checkout:

```bash
JUCE_ROOT=/absolute/path/to/JUCE bash BUILD_UNIVERSAL.command
```

## Install the generated build

After a successful build, unpack the generated installer ZIP and run its `install.command`. The installer writes only the `GNN4FOA_v2` AU/VST3 bundle names in the user plug-in folders.

For public distribution outside a development machine, replace ad-hoc signing with an appropriate Apple Developer ID signing/notarization workflow.

## Product identity

- Product: `GNN4FOA_v2`
- Version: `2.0.0`
- Bundle identifier: `it.polimi.ispl.GNN4FOAv2`
- Manufacturer code: `ISPL`
- Plug-in code: `G4v2`

The machine-facing product name deliberately contains no whitespace to avoid compiler response-file tokenization problems.

## Legacy Intel runtime contract

For this release, “legacy Mac” means a 64-bit Intel Mac capable of running macOS High Sierra 10.13. The generated x86_64 slice must not require AVX, AVX2, FMA, or a native-CPU build setting. The release gate also prevents future project edits from introducing `-march=native`, `-mcpu=native`, or explicit AVX/FMA flags.

The Apple matrix backend uses Accelerate/vDSP when available. Portable tests on a real macOS host therefore link `-framework Accelerate`; a scalar fallback remains compile- and run-tested independently with `GNN4FOA_FORCE_PORTABLE_GEMM=1`.
