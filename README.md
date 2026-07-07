# Timestamp Generator

timestamp_gen is an open-source Python tool for generating simulated single-photon timestamp data from rendered depth and normal maps.

This project converts VisionSIM/Blender rendered scenes into timestamp datasets that can be used for token-based ToF processing, event-style depth sensing experiments, and quantitative comparisons between timestamp-based and frame-based depth methods.

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
8. Optionally computes preprocessing data for downstream token processing.

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
* Saves timestamp datasets in a reusable .npz format with metadata.

## Pipeline

1. Obtain depth and normal EXR files.

This can be done using a python script to generate a Blender file. The resulting .blend file can then be processed through VisionSIM to output EXR files.

On a Windows system, this generation is done through wsl, since the backslash directory dividers that Windows uses is not supported by VisionSIM. 

2. Simulating timestamps

Once adding the EXR files to the timestamp_gen repo, you can change the user settings in timestamp_gen.py to use the desired folder.