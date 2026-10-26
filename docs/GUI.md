# Graphical interface

The GNN4FOA editor is a 1000 × 760 px interface with a spatial map on the left and controls/diagnostics on the right.

## Spatial map

Known microphone nodes are blue. The reconstructed/query node is orange. Coordinates are expressed in metres in a shared global frame.

The map also displays current node-weight links and a short history of recent relative weights. In GNN mode, the overlay exposes used, learned and Delaunay relative contributions; in Delaunay mode it shows the local geometric contribution.

### Tools

- **Fit View**: centres and scales the current scene.
- **Hand**: pans the map.
- **Node**: selects/moves microphone and query nodes.
- **Wall**: adds a line annotation.
- **Source**: adds a source marker annotation.
- **Dashed**: draws a dashed annotation.
- **Undo**: removes the latest annotation.
- **Lock Scene**: prevents edits to known microphones and annotations; the orange query remains movable.

Wall/source/dashed annotations are graphical only and are not published to the DSP.

### Navigation and editing

- Mouse wheel: zoom.
- Node mode + drag: move a node in X/Y.
- Shift + drag a blue microphone: change microphone yaw.
- Click the map, then Arrow keys: move the orange query at 1.35 m/s.
- Shift + Arrow keys: move at 0.35 m/s.
- Numeric X/Y/Z fields: precise coordinate entry.
- Microphone yaw field: precise local-to-global yaw in degrees.

The query node has position only and no orientation.

## Reconstruction controls

### Interpolation

`Delaunay Baseline` selects the local paper-faithful DirAC/Delaunay path. `GNN Learned` selects the learned graph model.

### FOA nodes

Manual active node count from 1 to 10. Selecting N consumes exactly the first 4N input channels (4, 8, 12, ..., 40). It is not inferred from track width or signal activity.

### GNN threshold

Relative energy threshold in dB. A bin enters the learned region at the selected threshold and exits only after falling 3 dB lower. Delaunay↔GNN transitions are crossfaded over two model hops (16 ms), so bins near the boundary do not chatter or hard-switch.

### DirAC covariance smoothing

Recursive smoothing coefficient used when estimating spatial covariance statistics and the resulting energy/intensity/diffuseness descriptors.

### GNN temporal average

Optional averaging of learned weights across recent learned frames. A value of `1` disables additional multi-frame averaging.

## Diagnostics

The diagnostic panel reports runtime state including:

- host/model sample rates and resampling state;
- active node/input count;
- model worker timing and deadline counters;
- GNN actual/predicted forward timing, inferred-bin count and adaptive stride;
- active learned-bin count and missing-coverage bin count;
- gate enter/exit counters and coverage-only/full inference counters;
- acoustic novelty-triggered refreshes;
- output reservoir, current queue, minimum observed queue and starvation state;
- learned raw/applied weight-sum statistics and clamp activity.

These values are intended for runtime and performance inspection and do not change the reconstruction by themselves.
