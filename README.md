# Timestamp Generator

**timestamp_gen** is an open-source Python pipeline for generating simulated single-photon time-of-flight (ToF) timestamps from Blender/VisionSIM depth and normal EXR files. The pipeline preserves multiple surfaces within each pixel, estimates interpolated scene geometry between rendered frames, and produces photon timestamps, mini-histograms, and depth estimates for studying dynamic scenes and low-latency sensing.

## Requirements

- Python 3.11.x
- Python dependencies listed in `requirements.txt`.
- OpenCV with OpenEXR support
- Blender for scene creation and rendering (the included cluster scripts reference 4.0.2)
- VisionSIM for rendering depth and normal EXRs; install it separately from the Python dependencies
- SLURM workload manager (HPC pipeline only)

## Installation

```bash
git clone https://github.com/GribDev1/timestamp_gen.git
cd timestamp_gen
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

In Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Windows workflow with WSL

This workflow uses Linux Blender and VisionSIM inside WSL to render EXRs, then runs timestamp generation and analysis in Windows. You can also keep the entire pipeline in Linux. A separate VirtualBox VM is not required when using WSL; VirtualBox is an alternative Linux environment that has not been verified here.

### 1. Set up the repository in WSL

Open your WSL terminal. If you have not already cloned the project there:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/GribDev1/timestamp_gen.git
cd timestamp_gen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing WSL checkout, start future sessions with:

```bash
cd ~/projects/timestamp_gen
source .venv/bin/activate
```

Install Linux Blender and VisionSIM separately and make their executables available in the WSL terminal. `requirements.txt` does not install either application. Use installation instructions for your chosen VisionSIM version, then check the available commands:

```bash
blender --version
visionsim --help
```

The WSL clone already contains `blender_scripts/`; copying that folder separately is unnecessary. If you edit scripts in a separate Windows checkout, synchronize those changes before rendering. To view the WSL checkout in Windows File Explorer, run `explorer.exe .` from the project directory in WSL.

### 2. Create a Blender scene

Run a scene generator from the repository root:

```bash
blender --background --factory-startup --python blender_scripts/create_drone_flyby_blend.py
```

This example saves `drone_flyby.blend` in the current project directory. For another scene, use its `create_<scene_name>_blend.py` script. Blender scripts execute in Blender's Python environment; activating a shell virtual environment does not install dependencies into Blender's bundled Python.

### 3. Render depth and normal EXRs

The following command uses the VisionSIM interface referenced by the repository's render worker, with a 160 × 160 output:

```bash
visionsim blender.render-animation drone_flyby.blend inputs/drone_flyby \
  --render-config.depths \
  --render-config.normals \
  --render-config.no-debug \
  --render-config.width=160 \
  --render-config.height=160 \
  --render-config.device-type=cpu \
  --render-config.no-use-denoising \
  --render-config.max-samples=16 \
  --render-config.adaptive-threshold=0.05 \
  --render-config.no-allow-skips
```

Check your installed VisionSIM version's help if its options differ. To use NVIDIA GPU rendering, replace `cpu` with `cuda` only when your Linux Blender/VisionSIM installation supports CUDA and detects the GPU.

Expected input folders after rendering are `inputs/drone_flyby/depths/` and `inputs/drone_flyby/normals/`.

### 4. Generate timestamps in Windows or WSL

To continue in Windows, copy the rendered `inputs/drone_flyby/` folder into the same relative location in your Windows repository checkout. Use a separate Windows virtual environment created with the installation steps above; do not reuse the WSL `.venv` on Windows.

Alternatively, stay in WSL and run the timestamp commands below from the WSL checkout without copying the EXRs. The generation and analysis examples use forward-slash paths that also work as Python arguments in PowerShell.

## Input requirements

- Provide at least two depth EXRs and the same number of corresponding normal EXRs.
- Depth and normal files are sorted separately by filename. Use matching, zero-padded names in time order.
- Use consistent image dimensions at least as large as the ToF grid, meter-valued depth, and normal maps with at least three channels.
- Match the rendering camera field of view to the selected sensor model; the generator uses that field of view to convert camera-axis depth to ray range.
- Set `--render-fps` to the actual frame rate used for rendering. Rendered image resolution and the `--render-dir` default name do not determine frame rate.

## Timestamp generation

The following options configure `timestamp_gen.py`. Run commands from the repository root:

| Argument | Description | Default |
| --- | --- | --- |
| `--sensor` | Sensor preset name from `configs/sensors/`. | `generic_spad_sensor` |
| `--render-dir` | Directory containing depths/ and normals/ subfolders. | full_render_320x160 |
| `--output-dir` | Directory where timestamp outputs will be saved. | timestamp_output |
| `--random-seed` | Random seed for reproducible timestamp simulation. | 0 |
| `--render-fps` | Frame rate of the rendered input sequence. | 240.0 |
| `--hist-bins` | Number of histogram bins for timestamp depth estimation | 32 |
| `--hist-depth-min` | Minimum histogram depth in meters. | 0.1 |
| `--hist-depth-max` | Maximum histogram depth in meters. | 20.0 |
| `--no-full-dataset` | Do not save full per-block timestamp dataset. | N/A |
| `--no-precomputed` | Do not save timestamp_precomputed.npz. | N/A |
| `--no-progress` | Disable the timestamp generation progress bar. | N/A |
| `--pixel-y` | Selected pixel row; supply together with `--pixel-x` and `--no-full-dataset`. | None |
| `--pixel-x` | Selected pixel column; supply together with `--pixel-y` and `--no-full-dataset`. | None |
| `--start-block` | First zero-based timestamp block to generate. | 0 |
| `--end-block` | Exclusive zero-based timestamp block at which to stop. | TO END |
| `--block-size` | Number of laser pulses per timestamp block. Overrides the sensor preset value. | 256 |

A full-scene example using the VL53L8CH-inspired grid and explicit histogram bounds:

```bash
python timestamp_gen.py --sensor vl53l8ch --render-dir inputs/drone_flyby --output-dir outputs/drone_flyby --render-fps 240 --block-size 256 --hist-bins 18 --hist-depth-min 0.02 --hist-depth-max 4.0
```

For a short trial, generate the first 100 blocks into a separate directory:

```bash
python timestamp_gen.py --sensor vl53l8ch --render-dir inputs/drone_flyby --output-dir outputs/drone_flyby_trial --render-fps 240 --start-block 0 --end-block 100
```

Partial runs save their precomputed data under `precomputed_shards/`. For a full-scene run with reduced disk use, add `--no-full-dataset`; this retains histogram/depth outputs but omits raw photon blocks and `metadata.json`.

Use `python timestamp_gen.py --help` for the complete interface. The implemented defaults above take precedence over inconsistent default descriptions in the current help text.

## Sensor presets

Presets live in `configs/sensors/`. Add a YAML file there and select its filename stem with `--sensor`.

| Preset | Grid | Valid range | Laser rate | YAML block size | Detection probability | Jitter standard deviation |
| --- | --- | --- | --- | --- | --- | --- |
| `generic_spad_sensor` | 32 × 64 | 0.01–10 m | 10 MHz | 256 | 0.05 | 0 s |
| `vl53l8ch` | 8 × 8 | 0.02–4 m | 10 MHz | 256 | 0.05 | 0 s |
| `big_px_tof` | 1 × 1 | 0.01–10 m | 10 MHz | 100,000 | 1.0 | 0 s |

Depth validity uses strict bounds. The current generator defaults `--block-size` to 256, overriding the YAML value even when the option is omitted. Reproducing that YAML block size requires `--block-size 100000`; first address the histogram count overflow described in Notes.

Histogram and readout sections in YAML are not applied automatically by the active sensor loader. Set histogram options on the command line. The `vl53l8ch` preset is a simulation approximation; it does not emulate the device's complete electronics or output interface.

## Simulation model

For each acquisition block, the generator estimates geometry between neighboring rendered frames. It linearly interpolates likely same-surface points and switches between frames near the interval midpoint for likely visibility changes. The same-surface decision uses a range-change threshold derived from 15 m/s and a normal cosine threshold of 0.97.

Within each ToF pixel footprint, ranges are sampled with replacement. Weighted sampling is enabled by `USE_WEIGHTED_DEPTH_SAMPLING = True` in `timestamp_gen.py`. Weights combine the nonnegative incidence cosine and inverse-square range; invalid weights fall back to uniform sampling. Change that constant to `False` to use uniform sampling. There is no command-line switch for this setting.

Clean timestamps are round-trip delays, `tau = 2 * range / c`. Bernoulli detection and Gaussian timing jitter produce noisy timestamps; missed detections or invalid geometry produce `NaN`. Detection probability is a fixed preset parameter: the sampling weights affect the chosen surface, not the Bernoulli detection rate.

Block duration is `L / laser_rate_hz`. At 10 MHz with 256 pulses, each block lasts 25.6 µs and has 12.8 expected detections per pixel with valid geometry at probability 0.05. Geometry is evaluated at block times rather than separately for every pulse within a block.

For `N` input frames at render rate `F`, the scene duration is `(N - 1) / F`. The number of complete acquisition blocks is the floor of scene duration divided by block duration; any trailing fraction is omitted.

Depth estimation finds the highest histogram bin and computes a count-weighted timestamp average over that bin and up to two neighbors on either side. Empty histograms produce `NaN` depth.

## Output format

A complete run with both output types enabled creates `metadata.json`, `frames/frame_000001.npz` and subsequent block files, plus `timestamp_precomputed.npz`.

Let `B` be the number of blocks, `L` the pulses per block, `H × W` the sensor grid, and `K` the histogram bin count.

| Per-block field | Shape / meaning |
| --- | --- |
| `frame_number` | One-based acquisition block number |
| `simulation_time_s` | Block end time in seconds |
| `sampled_depths_m` | `[L, H, W]` sampled ray ranges in meters |
| `timestamps_clean_s` | `[L, H, W]` ideal round-trip delays in seconds |
| `detection_mask` | `[L, H, W]` raw Bernoulli mask |
| `timestamps_noisy_s` | `[L, H, W]` measured delays; `NaN` where unavailable |
| `source_depth_file`, `source_normal_file` | Source tracking strings |

The raw detection mask can be true where geometry is invalid. Use finite values in `timestamps_noisy_s` to count usable detections.

| Precomputed field | Full-grid shape / meaning |
| --- | --- |
| `tof_depths` | `[B, H, W]` histogram depth estimates in meters |
| `all_I` | `[B, H, W]` finite timestamp fraction per block/pixel |
| `all_histograms` | `[B, H, W, K]` histogram counts |
| `tof_block_times_s` | `[B]` block end times |
| `hist_bin_centers_tau` | `[K]` bin centers in seconds |
| `hist_bin_centers_depth_m` | `[K]` bin centers in meters |
| `start_block`, `end_block` | Zero-based interval boundaries |

Load a complete full dataset with `TimestampDataset.load(...)`, or inspect the compact archive directly:

```python
import numpy as np
with np.load("outputs/drone_flyby/timestamp_precomputed.npz") as data:
    depths = data["tof_depths"]
    times_s = data["tof_block_times_s"]
    histograms = data["all_histograms"]
```

The full-dataset loader loads all block files into memory. For large runs, read individual NPZ files instead.

## Visualizations and CSV export

Create summary plots:

```bash
python -m analysis.visualize_timestamps --mode summary --input outputs/drone_flyby --output-dir outputs/drone_flyby/plots
```

Plot raw timestamps for pixel `(y=4, x=4)`; this requires saved full photon blocks:

```bash
python -m analysis.visualize_timestamps --mode timestamps --input outputs/drone_flyby --output-dir outputs/drone_flyby/plots --pixel-y 4 --pixel-x 4
```

Create an animated depth map:

```bash
python -m analysis.visualize_timestamps --mode depth-gif --input outputs/drone_flyby --output-dir outputs/drone_flyby/plots --gif-stride 100 --gif-fps 12
```

Additional modes include `pixel`, `valid-gif`, and `histogram-gif`. Options support time windows, timestamp-axis limits, vertical time markers, and time-to-contact calculations. Use the module's `--help` for details.

Export one pixel's block summaries:

```bash
python npz_to_csv.py outputs/drone_flyby --mode pixel-summary --pixel-y 4 --pixel-x 4 --output-dir outputs/drone_flyby/csv
```

Other CSV modes are `pixel-photons`, `all-pixel-summary`, and `frame-photons`. Photon exports require full block files; CSV export also requires `metadata.json`.

## Slurm / HPC workflow

The scripts assume the repository is at `$HOME/projects/timestamp_gen`, Python is in `.venv`, and Blender is at `$HOME/software/blender-4.0.2-linux-x64`. Update paths, resources, and site-specific Slurm settings before use. Create `logs/` before submitting jobs.

Submit these stages individually, waiting for each preceding job to finish successfully before submitting the next. These commands do not automatically chain the first three stages:

```bash
mkdir -p logs
sbatch shell_scripts/create_blend.slurm drone_flyby
sbatch shell_scripts/render_visionsim.slurm drone_flyby
sbatch shell_scripts/merge_visionsim.slurm drone_flyby
bash shell_scripts/submit_timestamp_gen.sh drone_flyby vl53l8ch
```

`create_blend.slurm` deletes existing rendered inputs, segments, and outputs for the selected scene. Use it only when regenerating that scene's data.

The timestamp submission script divides the timeline into 1,000-block tasks, allows up to eight concurrent workers, and submits the merge job with a success dependency. Change `BLOCKS_PER_TASK` and `MAX_CONCURRENT` in that script to tune scheduling.

To merge precomputed block shards manually:

```bash
python scripts/merge_timestamp_block_shards.py --output-dir outputs/drone_flyby
```

Use separate output directories for different simulation settings. The generator also supports single-pixel output under `pixels/`, but the current `merge_timestamp_pixels.py` expects pixel-coordinate fields that the generator does not write; that merge workflow needs alignment before use.

## Repository structure

| Path | Purpose |
| --- | --- |
| `timestamp_gen.py` | Simulation CLI and photon/histogram generation |
| `timestamp_dataset.py` | Dataset classes, metadata, saving, and loading |
| `npz_to_csv.py` | CSV exports |
| `configs/sensor_presets.py` | Active YAML sensor loader and ray geometry |
| `configs/tof_sensor.py` | Separate sensor model implementation |
| `configs/sensors/` | Sensor preset YAML files |
| `configs/drone_paths/` | Drone trajectory configurations |
| `blender_scripts/` | Scene creation and render-segment helpers |
| `analysis/` | Plots, GIFs, geometry extraction, and motion/TTC analysis |
| `scripts/` | Timestamp archive merge utilities |
| `shell_scripts/` | Slurm jobs and pipeline orchestration |

## Notes

- Full photon datasets can be large. Reduce the block interval while developing, or use `--no-full-dataset` when raw timestamps are unnecessary.
- Precomputed arrays accumulate in memory before saving. Block shards reduce the size of each task.
- Histogram limits are separate from valid geometry limits; timestamps outside histogram bounds do not contribute to depth estimation.
- If EXR reads fail, confirm that your OpenCV build supports OpenEXR. The generator enables `OPENCV_IO_ENABLE_OPENEXR` before importing OpenCV.
- The current histogram arrays use `uint16` counts. Very large blocks, including the 100,000-pulse big-pixel preset, can overflow a bin above 65,535 counts; use smaller blocks or update the count dtype before such experiments.
- A fixed seed reproduces a run with the same processing layout. Changing block partitioning or switching between full-grid and single-pixel generation changes random-number consumption.
- The model uses geometry, incidence weighting, fixed detection probability, and timing jitter. It does not explicitly simulate ambient light, dark counts, detector dead time, multipath, or material-dependent optical transport.

## License

MIT License. See [LICENSE](LICENSE).

## Review scope

The command-line defaults, sensor presets, and workflow behavior were checked against repository code on October 13, 2026. Blender/VisionSIM rendering and cluster execution were not run during this documentation review.
