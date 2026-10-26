# Building GNN4FOA on macOS

## Requirements

- full Xcode 12.4 or newer;
- Git, unless JUCE is supplied through `JUCE_ROOT`;
- enough disk space to clone JUCE 9.0.1 and build Universal AU/VST3 binaries.

## One-command build

From the repository root:

```bash
bash BUILD_UNIVERSAL.command
```

The script:

1. validates Xcode and the generated Xcode project;
2. locates `ThirdParty/JUCE`, uses `JUCE_ROOT`, or clones JUCE tag `9.0.1`;
3. cleans only project-local build products;
4. builds AU and VST3 for `x86_64 + arm64`;
5. requests macOS deployment target 10.13;
6. checks architecture slices and deployment metadata;
7. rejects CBLAS/NEWLAPACK-linked learned backends;
8. ad-hoc signs the local build;
9. creates `GNN4FOA_v1_macOS_Universal_installer.zip`.

To use a local JUCE checkout:

```bash
JUCE_ROOT=/absolute/path/to/JUCE bash BUILD_UNIVERSAL.command
```

## Install the generated build

After a successful build, unpack the generated installer ZIP and run its `install.command`. The installer writes only the `GNN4FOA_v1` AU/VST3 bundle names in the user plug-in folders.

For public distribution outside a development machine, replace ad-hoc signing with an appropriate Apple Developer ID signing/notarization workflow.

## Product identity

- Product: `GNN4FOA_v1`
- Version: `1.0.0`
- Bundle identifier: `it.polimi.ispl.GNN4FOAv1`
- Manufacturer code: `ISPL`
- Plug-in code: `G4v1`

The machine-facing product name deliberately contains no whitespace to avoid compiler response-file tokenization problems.
