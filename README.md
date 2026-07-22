# Timestamp Generator

*timestamp_gen* is an open-source Python tool for generating simulated single-photon timestamp data from rendered depth and normal maps.

This project converts VisionSIM/Blender rendered scenes into timestamp datasets that can be used for downstream ToF research, event-style depth sensing experiments, and quantitative comparisons between timestamp-based and frame-based depth methods.

## Overview

The timestamp generator takes rendered depth and normal EXR files and simulates a single-photon ToF sensor pipeline.

For each rendered frame, the tool:

1. Loads high-resolution depth and normal maps.
2. Divides the image into a lower-resolution ToF sensor grid.
3. Samples sub-pixel depth values inside each ToF pixel footprint.
4. Converts sampled depths into clean photon timestamps.
5. Applies a Bernoulli detection model to simulate missed photons.
6. Adds Gaussian timing jitter to detected timestamps.
7. Saves clean and noisy timestamp blocks.
8. Optionally computes timestamp-derived diagnostic data such as histograms, depth estimates, and valid detection fraction.

The goal is to bridge rendered depth scenes and timestamp-based ToF algorithms.

## Features

* Converts rendered depth maps into photon timestamp streams.
* Supports clean and noisy timestamp models.
* Simulates missed detections using a Bernoulli detection probability.
* Adds Gaussian timing jitter.
* Supports weighted depth sampling using surface normals and distance falloff.
* Preserves sub-pixel depth mixtures at object boundaries and occlusions.
* Supports interpolated visibility switching between rendered frames.
* Computes timestamp-derived diagnostic data:
  * per-frame mini-histograms
  * block-rate depth estimates
  * valid detection fraction I
  * histogram bin centers
* Saves timestamp datasets in a reusable `.npz` format with metadata.

### Installing WSL (Windows users)

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

### 1. Creating Python Blender scripts

In this directory, there is a folder named blender_scripts. In this folder, there are multiple examples of python scripts to generate Blender files for rendering use.

To help simplify the process, I created a scene_builder class so you can add simple elements to your Blender files. For reference, I recommend looking that the example files included in this repository.

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
visionsim blender.render-animation INPUT_FILE.blend output/OUTPUT_RENDERED_EXR_FILES   --render-config.depths   --render-config.normals   --render-config.no-debug   --render-config.width=160   --render-config.height=160   --render-config.device-type=cuda   --render-config.no-use-denoising   --render-config.max-samples=16   --render-config.adaptive-threshold=0.05   --render-config.no-allow-skips 
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
| `--interpolation-steps` | Sets how many timestamp blocks are generated between each rendered frame pair. |
| `--no-interpolation` | Disables interpolated visibility switching. |
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

### `metadata.json`

Stores simulation settings such as:

* ToF sensor resolution
* block size
* laser rate
* detection probability
* timing jitter
* effective frame rate
* valid depth range
* timestamp units
* depth units

### Per-frame `.npz` files

Each frame file contains:

| Array | Shape | Description |
|---|---|---|
| `sampled_depths_m` | `[L, H, W]` | Sampled geometric ranges in meters. |
| `timestamps_clean_s` | `[L, H, W]` | Ideal timestamps before missed detections and jitter. |
| `detection_mask` | `[L, H, W]` | Boolean mask showing detected photons. |
| `timestamps_noisy_s` | `[L, H, W]` | Noisy detected timestamps, with `NaN` for missed detections. |


### `timestamp_precomputed.npz`

The optional precomputed file contains timestamp-derived diagnostic arrays:

| Array	| Description |
|---|---|
| `tof_depths` |	Block-rate depth estimates from timestamp histograms. | 
| `all_I` |	Valid detection fraction per pixel. |
| `all_histograms` | Per-frame mini-histograms. |
| `hist_bin_centers_tau` | Histogram bin centers in timestamp units. |
| `hist_bin_centers_depth_m` | Histogram bin centers converted to depth. |