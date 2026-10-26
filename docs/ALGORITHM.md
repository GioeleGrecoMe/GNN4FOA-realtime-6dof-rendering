# Reconstruction algorithms

## Signal convention

Each known microphone is a first-order Ambisonic signal ordered as `[W, Y, Z, X]`. Microphone yaw rotates local FOA directions into the common global frame before spatial analysis.

The model domain operates at 16 kHz with a 512-sample STFT and 128-sample hop.

## DirAC analysis

For every active microphone and time-frequency bin, the processor estimates energy, active-intensity direction and diffuseness from smoothed FOA covariance statistics. These descriptors are used by both reconstruction modes.

## Delaunay / DirAC baseline

For three or more microphones, a deterministic 2-D Delaunay triangulation is built from active microphone X/Y coordinates.

### Interior query

The triangle containing the query is selected. Its three barycentric weights are non-negative and sum to one; every other microphone receives zero geometric weight.

For weights `g_m`:

```text
E_hat   = sum_m g_m E_m
v_hat   = sum_m g_m (i_m / E_m)
        = sum_m g_m (1 - psi_m) u_m
psi_hat = 1 - ||v_hat||
u_hat   = v_hat / ||v_hat||    when ||v_hat|| > 0
```

This is intentionally different from directly averaging diffuseness.

### Exterior query

The query position is projected to the nearest point on the triangulated region boundary. Barycentric interpolation is then evaluated at that projected point. The method does not perform all-node inverse-distance extrapolation.

### N = 2 extension

The referenced triangular method requires at least three microphones. GNN4FOA supports `N=2` using the one-dimensional analogue: linear interpolation on the segment connecting the two microphones, clamped to the nearest endpoint outside the segment.

### Waveform / phase prototype

Pure Delaunay mode uses the FOA recording from the microphone nearest to the actual query position as the waveform/phase prototype. Spatial parameters are adapted without directly summing widely separated FOA phases. A short phase-aligned transition is used when the closest prototype changes.

Reference:

A. Politis, L. Pajunen, J. Leppänen, S. Mate, A. Eronen, “Wide-Area 6DoF Rendering of Multi-Point Ambisonic Recordings Based on Interpolation of Spatial Parameters,” WASPAA 2023, DOI: 10.1109/WASPAA58266.2023.10248142.

## GNN learned path

The learned renderer forms a complete directed graph over `N` known microphone nodes and one query node.

The embedded network uses two message-passing stages with hidden dimension 48. Node and edge features combine spatial coordinates with DirAC-like acoustic descriptors. A shared readout predicts positive interpolation weights for known nodes.

The model may assign non-zero weight to any active known microphone. Relative learned ratios are retained. If the raw positive weights sum to `S`, the live interface applies:

```text
S_target = clamp(S, 0.5, 2.0)
w_applied[m] = w_raw[m] * S_target / S
```

Therefore the learned sum is not forced to one.

Low-energy bins use the Delaunay baseline as a reliability fallback. A previously learned high-energy bin retains learned history under temporary compute pressure; fresh threshold re-entry invalidates stale learned history before reuse.

## Adaptive learned inference

STFT/DirAC processing remains frame-synchronous. GNN inference may run at an adaptive stride according to measured model cost and output-reservoir safety. Cadence is based on processed audio/model frames rather than wall-clock time, so anticipative/offline host processing does not stretch the learned update interval relative to the audio timeline.
