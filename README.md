# Timestamp Generator

`timestamp_gen` is an open-source Python pipeline for generating simulated
single-photon direct time-of-flight data from Blender and VisionSIM renders.

The project converts high-resolution depth and surface-normal animations into
pulse-rate photon timestamps, block-rate depth estimates, histograms, detection
statistics, and motion-related diagnostic data.

It is intended for:

- dynamic direct-ToF research
- SPAD timestamp simulation
- frame-based versus timestamp-based sensing comparisons
- low-latency depth and motion experiments
- occlusion and multi-surface analysis
- drone and robotics simulations
- downstream token-based or event-driven ToF processing

## Overview

The pipeline has four main stages:

1. Create an animated Blender scene.
2. Render depth and normal EXR sequences using VisionSIM.
3. Simulate single-photon ToF timestamp blocks.
4. Visualize or process the resulting timestamp data.

For each ToF acquisition block, the timestamp generator:

1. Loads the two rendered frames surrounding the block time.
2. Determines the scene state at the block acquisition time.
3. Interpolates same-surface geometry between rendered frames.
4. Uses visibility switching for likely occlusion or surface changes.
5. Maps rendered camera rays into a lower-resolution ToF sensor grid.
6. Samples sub-pixel geometric ranges within each ToF pixel.
7. Optionally weights samples using surface incidence and inverse-square
   distance falloff.
8. Converts range measurements into round-trip photon timestamps.
9. Applies Bernoulli missed-detection sampling.
10. Applies Gaussian timing jitter.
11. Builds a mini-histogram for each ToF pixel.
12. Computes block-rate depth and valid-detection estimates.
13. Saves raw and/or precomputed results.

## Features

### Scene generation

- Reusable Blender scene-building utilities.
- Camera position and rotation animation.
- Plane, cube, cylinder, sphere, and custom wavy-surface geometry.
- Linear and constant keyframe interpolation.
- Support for Blender 4.x and newer animation APIs.
- Example drone flyby, wall-approach, and landing scenes.

### VisionSIM rendering

- Depth EXR rendering.
- Surface-normal EXR rendering.
- Configurable render resolution and frame rate.
- CUDA rendering where supported.
- Headless rendering for WSL, Linux, and Slurm environments.

### Timestamp simulation

- Configurable ToF sensor resolution.
- Configurable laser repetition rate and pulses per block.
- Continuous generation at the ToF block rate.
- Clean round-trip timestamp generation.
- Bernoulli photon-detection model.
- Gaussian timing jitter.
- Sub-pixel surface-mixture preservation.
- Range correction using rendered camera-ray direction.
- Surface-normal and inverse-square distance weighting.
- Same-surface temporal interpolation.
- Hard visibility switching for likely occlusions.
- Reproducible simulation using a random seed.

### Precomputed diagnostics

- Mini-histograms for every ToF pixel and acquisition block.
- Histogram-centroid depth estimates.
- Valid photon-detection fraction.
- Timestamp and depth histogram bin centers.
- Explicit ToF block timing.

### Visualization and analysis

- Depth maps at selected blocks.
- Valid-detection fraction maps.
- Pixel histograms.
- Mean depth and valid-detection fraction over time.
- Raw photon timestamp-versus-simulation-time plots.
- Animated depth, detection-fraction, and histogram GIFs.
- Radial velocity and closing-speed estimates.
- Time-to-contact maps, plots, CSV files, and NPZ output.
- Blender-camera and ToF-sensor FoV overlap visualization.
- Top-row and bottom-row ToF-zone ray fans.
- Per-rendered-ray expected photon-contribution maps.


## Requirements

The pipeline is primarily intended for Linux or WSL and requires:

- Python 3.10 or newer
- Blender
- VisionSIM
- OpenCV with OpenEXR support
- NumPy
- Matplotlib
- tqdm
- PyYAML
- Pillow

Optional:

- CUDA-capable GPU for VisionSIM rendering
- Slurm for cluster execution

## Clone and create the Python environment

```bash
git clone <repository-url>
cd timestamp_gen

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Installing WSL (Windows users)

The entire pipeline depends on being able to use a bash-based terminal. My preferred method is to use Windows Subsystem for Linux (wsl). Most desktops are able to do this by opening a terminal and entering the following commands:

```powershell
wsl --install -d Ubuntu
```

If this doesn't work for whatever reason, run the following commands and it should work:

```powershell
wsl --update --web-download
wsl --install --web-download -d Ubuntu
```

## General Pipeline

## 1. Create an animated Blender scene

Example Blender scene scripts are located in `blender_scripts/`.

The shared `scene_builder.py` module includes helpers for:

- configuring resolution, frame rate, and frame range
- adding cameras and lights
- adding basic geometry
- creating custom wavy landing surfaces
- animating object or camera position
- animating object or camera rotation
- selecting linear or constant interpolation
- saving the generated `.blend` file

Generate a scene using Blender in background mode:

```bash
blender --background --python blender_scripts/create_TEST_NAME_blend.py
```

### 2. Rendering animations using VisionSIM (depth EXR and normal EXR files)

Since I am working on a Windows operating system, using a traditional PowerShell does not support VisionSIM. This is mainly due to the backslashes that separate  directories on Windows. My solution was to bring the pipeline to this point using Windows Subsystem for Linux (wsl). After installing, you can launch wsl by running the follow command in any terminal:

```bash
wsl
```

When running for the first time, you have to create a directory for your animations. For example,

```bash
mkdir ~/render_visionsim
```

After moving into that directory, 

```bash
cd ~/render_visionsim
```

you can create a python virtual environment to run the python scripts to generate Blender scenes.

```bash
python3 -m venv .venv 
```

```bash
source .venv/bin/activate 
```

To generate `.blend` files, make sure you have a Python Blender script in the wsl directory and run the following (using your own file name):

```bash
blender --background --python create_example_blend.py
```

To render the animation via VisionSIM, you can then run the following command in wsl:

```bash
visionsim blender.render-animation INPUT_FILE.blend inputs/OUTPUT_RENDERED_EXR_FILES   --render-config.depths   --render-config.normals   --render-config.no-debug   --render-config.width=160   --render-config.height=160   --render-config.device-type=cuda   --render-config.no-use-denoising   --render-config.max-samples=16   --render-config.adaptive-threshold=0.05   --render-config.no-allow-skips 
```

From this point, you can copy the output folder from the previous command and paste it into the *timestamp_gen* directory.

### 3. Simulating timestamps

Once the rendered EXR files are available, place the render folder in the `timestamp_gen` repository or provide the path to it from the command line.

There are also a set of command-line methods to change the output:

### Common command-line options

| Option | Description |
|---|---|
| `--sensor` | Selects a sensor preset from `configs/tof_sensors.csv`. |
| `--render-dir` | Path to the folder containing `depths/` and `normals/`. |
| `--output-dir` | Path where timestamp outputs will be saved. |
| `--random-seed` | Sets the random seed for reproducible simulations. |
| `--render-fps` | Sets the frame rate of the rendered input sequence. |
| `--hist-bins` | Sets the number of timestamp histogram bins. |
| `--hist-depth-min` | Sets the minimum histogram depth in meters. |
| `--hist-depth-max` | Sets the maximum histogram depth in meters. |
| `--adaptive-gate-m` | Sets the adaptive time-gate half-width in meters. |
| `--no-full-dataset` | Skips saving the full per-frame timestamp dataset. |
| `--no-precomputed` | Skips saving `timestamp_precomputed.npz`. |

If you prefer to see this table in the terminal window, you can use the following command:

```powershell
python timestamp_gen.py --help
```

For an example of a command:

```powershell
python timestamp_gen.py --sensor vl53l8ch 
--render-dir inputs/OUTPUT_RENDERED_EXR_FILES --output-dir outputs/TEST_NAME 
--render-fps 240
```

## Input Format

The timestamp generator expects a render folder containing `depths/` and `normals/` subfolders:

```text
full_render_320x160/
├── depths/
│   ├── frame_000001.exr
│   ├── frame_000002.exr
│   └── ...
└── normals/
    ├── frame_000001.exr
    ├── frame_000002.exr
    └── ...
```

Depth files should contain depth values in meters.

Normal files should contain 3-channel surface normal data. OpenCV loads EXR channels in BGR order, so the script converts normal maps to RGB internally.

## Output Format

The output directory contains:

```text
timestamp_output/
├── metadata.json
├── timestamp_precomputed.npz
└── frames/
    ├── frame_000001.npz
    ├── frame_000002.npz
    └── ...
```

# Metadata

### `metadata.json`

The metadata file stores:

- ToF sensor height and width
- laser repetition rate
- pulses per block
- ToF block rate
- ToF block duration
- detection probability
- timing-jitter standard deviation
- minimum and maximum valid range
- timestamp and depth units
- missed-detection representation
- whether normal- and distance-weighted sampling was enabled

## Timestamp-generation command-line options

| Option | Description |
|---|---|
| `--sensor` | Sensor YAML preset name from `configs/sensors/`. |
| `--render-dir` | Directory containing `depths/` and `normals/`. |
| `--output-dir` | Directory where generated files are saved. |
| `--random-seed` | Seed for reproducible random timestamp generation. |
| `--render-fps` | Frame rate used when the EXR animation was rendered. |
| `--hist-bins` | Number of bins in each per-pixel mini-histogram. |
| `--hist-depth-min` | Minimum histogram depth in meters. |
| `--hist-depth-max` | Maximum histogram depth in meters. |
| `--no-full-dataset` | Skip saving individual raw timestamp block files. |
| `--no-precomputed` | Skip saving `timestamp_precomputed.npz`. |
| `--no-progress` | Disable the progress bar. |

### `timestamp_precomputed.npz`

| Array | Shape | Description |
|---|---:|---|
| `tof_depths` | `[B,H,W]` | Histogram-derived depth estimate for each ToF block. |
| `all_I` | `[B,H,W]` | Valid detection fraction for each pixel and block. |
| `all_histograms` | `[B,H,W,N]` | Mini-histogram counts. |
| `tof_block_times_s` | `[B]` | Acquisition time associated with each block. |
| `hist_bin_centers_tau` | `[N]` | Histogram bin centers in timestamp seconds. |
| `hist_bin_centers_depth_m` | `[N]` | Histogram bin centers converted to meters. |

Where:

- `B` is the number of timestamp blocks
- `L` is the number of laser pulses per block
- `H × W` is the ToF sensor resolution
- `N` is the number of histogram bins

## Visualizing timestamp results

Generate standard diagnostic plots:

```bash
python visualize_timestamps.py \
    --input outputs/drone_flyby/timestamp_precomputed.npz \
    --output-dir outputs/drone_flyby/figures \
    --pixel-y 4 \
    --pixel-x 4
```

# Miscellaneous

## Dataset size and performance

Timestamp datasets can become very large.

For a sensor with:

- \(H \times W\) pixels
- \(L\) pulses per block
- \(B\) blocks

the raw number of timestamp samples is:

\[
N = B L H W
\]

For a 10 MHz laser with 256 pulses per block, the block rate is approximately
39.1 kHz. Even a one-second 8 × 8 simulation contains:

\[
39{,}062 \times 256 \times 8 \times 8
\approx 640\text{ million samples}
\]

Use `--no-full-dataset` when only histogram, depth, or valid-detection outputs
are required.