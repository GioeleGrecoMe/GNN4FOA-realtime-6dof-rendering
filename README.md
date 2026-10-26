<p align="center">
    <img src="docs/images/GNN4FOA.png" alt="alt text">
</p>

# 

**Real-time 2-D FOA sound-field reconstruction from multiple Ambisonic recordings, using either a paper-faithful DirAC-Delaunay baseline or a learned Graph Neural Network.**

Developed at the **Image and Sound Processing Lab (ISPL), Politecnico di Milano**.

**Authors:** Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

## Overview

GNN4FOA is a JUCE audio plug-in for reconstructing a first-order Ambisonic (FOA) signal at a movable query position from up to ten known FOA recordings measured at different positions.

The plug-in exposes two reconstruction modes:
<table>
  <tr>
    <td width="85%" valign="middle">
      • <strong>Delaunay Baseline</strong>: local DirAC parameter interpolation based on a 2-D Delaunay triangulation. Inside the measurement region, only the three microphones forming the triangle that contains the query position are used. Outside the region, the query is projected to the closest triangulation boundary. The waveform/phase prototype is taken from the closest FOA microphone.
    </td>
    <td width="15%" align="center" valign="middle">
      <img src="docs/images/Comparing 1.png" width="100%">
    </td>
  </tr>
  <tr>
    <td width="85%" valign="middle">
      • <strong>GNN Learned</strong>: a graph neural network that uses all manually active FOA nodes and their spatial/acoustic features to predict positive interpolation weights. The relative learned ratios are preserved, while the total learned weight sum is constrained to <strong>[0.5, 2.0]</strong>.
    </td>
    <td width="15%" align="center" valign="middle">
      <img src="docs/images/Comparing 2.png" width="100%">
    </td>
  </tr>
</table>

The four reconstructed outputs remain in a **global FOA frame**. The plug-in does not perform binaural/HRTF decoding or listener-head rotation.

## Audio I/O

The plug-in has a fixed host topology:

- **40 input channels** = up to 10 FOA nodes
- **4 output channels** = reconstructed FOA
- FOA channel convention: **`[W, Y, Z, X]`**

The active number of nodes is selected manually in the GUI. No host-channel auto-detection or runtime bus renegotiation is used.

| FOA node | Host input channels |
|---|---|
| Node 1 | 1–4 |
| Node 2 | 5–8 |
| Node 3 | 9–12 |
| Node 4 | 13–16 |
| Node 5 | 17–20 |
| Node 6 | 21–24 |
| Node 7 | 25–28 |
| Node 8 | 29–32 |
| Node 9 | 33–36 |
| Node 10 | 37–40 |
| Reconstructed FOA | outputs 1–4 |

If `N` nodes are selected, the DSP consumes exactly the first `4N` input channels. Remaining input channels are ignored.

## Installation
1. Unzip the installer.
2. Run the `installer.command` file by double-clicking over it.
3. Run the `verify.command` as a check.

## Quick start in REAPER
1. Create a multichannel track able to carry the required FOA signals.
2. Route each known FOA recording to one consecutive 4-channel group using the mapping above.
3. Insert **`GNN4FOA_v2`** on the track.
4. Set **FOA nodes** to the number of recordings actually routed to the plug-in.
5. On the map, set the position of each blue microphone node. Set microphone yaw when the local FOA frame is not aligned with the global frame.
6. Set the orange query/listener position.
7. Choose **Delaunay Baseline** or **GNN Learned**.
8. Route the four plug-in outputs as the reconstructed FOA signal.

For controlled experiments, verify channel order and node positions before comparing the two interpolation modes. A wrong node-to-audio association affects the learned model especially strongly because acoustic features are part of the GNN input.

## Interface
The editor is divided into a spatial map and a control/diagnostic panel.

### Spatial map

- **Blue nodes** — known FOA microphone positions.
- **Orange node** — reconstructed/query position.
- **Weight links** — line thickness and labels visualize the contribution of each active node.
- **Weight history** — approximately 20 seconds of recent relative node-weight evolution.
- **Global axes** — show the global +X / +Y reference used by position and microphone yaw.

Map controls:

- **Fit View** — fits the current scene into the map.
- **Hand** — pans the map.
- **Node** — selects and moves nodes.
- **Wall / Source / Dashed** — display-only annotations; they never enter the audio DSP.
- **Undo** — removes the latest annotation.
- **Lock Scene** — freezes known microphone positions and annotations while still allowing the orange query to move.
- **Mouse wheel** — zoom.
- **Shift + drag a blue microphone in Node mode** — changes its yaw.
- **Arrow keys** — move the orange query at 1.35 m/s after clicking the map.
- **Shift + Arrow keys** — precision movement at 0.35 m/s.

### Main controls

| Control | Meaning |
|---|---|
| **Interpolation** | Switches between paper-faithful Delaunay/DirAC and the learned GNN. |
| **FOA nodes** | Manual active node count, 1–10. This is the only authority for `N`. |
| **GNN threshold** | Relative energy threshold. GNN entry uses the selected threshold; exit has 3 dB hysteresis and a 16 ms bidirectional transition to avoid gate chatter/clicks. |
| **DirAC covariance smoothing** | Recursive smoothing used for energy, intensity direction and diffuseness estimation. |
| **GNN temporal average** | Optional temporal averaging of learned weights. `1` gives the least additional smoothing. |
| **Node settings** | Numeric X/Y/Z coordinates and microphone yaw in degrees. The orange query has no yaw. |
| **Diagnostics** | Model timing, active bins, inference stride, worker/reservoir status, FIFO/starvation counters and learned weight statistics. |

A more detailed guide is available in [`docs/GUI.md`](docs/GUI.md).

## Reconstruction methods

### Paper-faithful DirAC / Delaunay baseline
For `N >= 3`, active microphone positions are triangulated in 2-D. For a query inside the measurement region, the single containing Delaunay triangle is selected and only its three vertices receive non-zero barycentric weights.

For geometric weights $g_m$, the reconstructed spatial parameters follow:

$\hat{E}   = \sum_m{g_m E_m}$

$\hat{\mathbf{v}}   = \sum_m{g_m \frac{\mathbf{i}_m}{E_m}} = \sum_m{g_m (1 - \psi _m) \mathbf{u}_m}$

$\hat{\psi} = 1 - \lVert\hat{\mathbf{v}}\rVert$

$\hat{\mathbf{u}}   = \frac{\hat{\mathbf{v}}} {\lVert\hat{\mathbf{v}}\rVert} \quad \quad   when \quad \lVert\hat{\mathbf{v}}\rVert > 0$


Diffuseness $\hat{\psi}$ is therefore derived from the magnitude of the interpolated normalized-intensity vector $\hat{\mathbf{v}}$ rather than averaged directly. In pure Delaunay mode, the waveform/phase prototype is the FOA microphone closest to the query, avoiding direct phase mixing between widely spaced recordings.

The baseline follows:

> A. Politis, L. Pajunen, J. Leppänen, S. Mate, A. Eronen, “Wide-Area 6DoF Rendering of Multi-Point Ambisonic Recordings Based on Interpolation of Spatial Parameters,” *WASPAA 2023*, DOI: [10.1109/WASPAA58266.2023.10248142](ieeexplore.ieee.org/document/10248142/).

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for implementation details and the explicit N=2 extension.

### GNN learned interpolation
The learned path builds a complete directed graph from the active known microphones plus the query node. It uses spatial positions together with DirAC-like acoustic descriptors and predicts positive per-node interpolation weights.

Key properties of the embedded model/runtime:

- up to 10 known FOA nodes + 1 query node;
- embedded checkpoint trained with exactly five known FOA nodes; `N=1..4` and `N=6..10` are graph-cardinality OOD extrapolations;
- message aggregation is scaled by `5/N` to preserve the training-degree magnitude;
- two message-passing layers;
- hidden dimension 48;
- positive learned weights;
- learned total weight sum constrained to **[0.5, 2.0]** without changing node-to-node ratios;
- compact inference on active frequency bins;
- 3 dB hysteretic low-energy Delaunay fallback with 16 ms Delaunay↔GNN parameter crossfades;
- missing-bin coverage refreshes infer only newly uncovered bins rather than the full active spectrum;
- bin-count-aware admission and audio-time scheduling with reservoir protection;
- macOS dense matrix products through Accelerate `vDSP_mmul`.

## Sample rate and real-time architecture
The model operates internally at **16 kHz** with a 512-sample STFT and 128-sample hop (125 TF frames/s). At other host sample rates, selected input FOA channels are resampled to 16 kHz before analysis and the reconstructed four-channel FOA output is converted back to the host rate.

The GNN does **not** run in the DAW audio callback. A dedicated model worker performs resampling, STFT/DirAC analysis, learned inference and synthesis. The callback uses preallocated FIFO/copy operations.

## Building on macOS

Requirements:

- full Xcode **12.4 or newer**;
- Git, unless a compatible JUCE tree is supplied through `JUCE_ROOT`;
- macOS build host.

From the repository root:

```bash
bash BUILD_UNIVERSAL.command
```

The build script uses **JUCE 9.0.1**, cloning that exact tag into `ThirdParty/JUCE` when required. It builds AU and VST3 as Universal `x86_64 + arm64` binaries with deployment target macOS 10.13, performs compatibility checks, and creates a local installer archive. The Intel slice is deliberately kept on a conservative x86_64 baseline: the build does not opt into SSE3/SSE4.x/AVX extensions and post-build validation rejects AVX/VEX instructions, so 2010-era Intel Macs that can run macOS High Sierra are not silently excluded by a modern build machine.

**Important:** a 2010 Mac is a supported *runtime target*, not a supported JUCE 9 build host. Build the Universal package with a newer Mac/Xcode, then install the resulting x86_64 slice on the legacy machine. The legacy runtime floor for this release is macOS 10.13.

To use an existing JUCE tree:

```bash
JUCE_ROOT=/path/to/JUCE bash BUILD_UNIVERSAL.command
```

See [`docs/BUILDING.md`](docs/BUILDING.md) for details.

## Tests

Portable algorithm and policy tests can be run on a C++17 environment with:

```bash
bash Tests/portable/run_all.sh
```

The suite covers, among other things, dynamic node counts, paper-faithful Delaunay geometry, learned weight-sum bounds, compact-bin equivalence, multi-node audio ingress, temporal safety, scheduling, continuity policies, the forced portable GEMM fallback and a simulated Apple/vDSP code path. The build command itself uses the shorter fail-fast `Tests/portable/run_release_gate.sh`; exhaustive performance sweeps are available with `bash Tests/portable/run_extended.sh`.

## Repository layout

```text
Source/                 audio engine, plug-in processor/editor, embedded GNN weights
Source/Eigen/           Eigen headers used by the model implementation
Builds/MacOSX/          generated Xcode project
JuceLibraryCode/        generated JUCE wrapper sources
Tests/portable/         portable regression tests and benchmarks
docs/                   user, algorithm, build and validation documentation
ThirdParty/              JUCE is cloned here by the build script
BUILD_UNIVERSAL.command macOS AU/VST3 build and packaging script
GNN4FOA.jucer           JUCE/Projucer project
```

## Authors and affiliation

- **Gioele Greco**
- **Filippo Marri**
- **Mirco Pezzoli**
- **Fabio Antonacci**

**Image and Sound Processing Lab (ISPL)**  
Dipartimento di Elettronica, Informazione e Bioingegneria  
**Politecnico di Milano**

Project website / lab: <https://ispl.deib.polimi.it/>

See [`AUTHORS.md`](AUTHORS.md) and [`CITATION.cff`](CITATION.cff).

## Third-party software and licensing

Eigen headers are vendored for the GNN runtime; JUCE 9.0.1 is acquired at build time. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependency notices.

This source release intentionally does **not** invent a project-wide license on behalf of the copyright holders. Add the intended top-level license before using GitHub to grant public redistribution/modification rights.


## Release

This repository corresponds to **GNN4FOA v2.0.0** (`GNN4FOA_v2` in DAW plug-in lists). See [`CHANGELOG.md`](CHANGELOG.md).
