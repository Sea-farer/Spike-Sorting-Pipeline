# CA1 Tetrode Place Maps

A Python-based spike sorting and spatial mapping pipeline for hippocampal CA1 recordings collected on the Axona system. This project automates the processing of raw tetrode data into interpretable neural and behavioral maps.

## Overview
This pipeline takes in raw electrophysiology and position-tracking data from Axona tetrode recordings and outputs:

- Spike-sorted units (via MClust cut files or direct tetrode inputs)

- Rate and occupancy-normalized place maps for individual neurons

- Visualization notebooks showing spatial firing patterns across sessions and subjects

- The code was developed to support computational neuroscience research on hippocampal place cells and memory circuits.

## Environment

Create a dedicated environment (Python 3.10 recommended):

```
conda create -n ca1maps python=3.10 -y
conda activate ca1maps
pip install -r ca1-tetrode-placemaps/requirements.txt
```

## Notebooks

- notebooks/01_quickstart.ipynb — minimal end-to-end walkthrough
- notebooks/02_spatial_maps.ipynb — full session coverage and rate maps
- notebooks/03_spatial_maps2.ipynb — same pipeline pointed to a different subject/date

Tip: In the first cell, set or verify `session_dir`, then Run All.

## Data layout

```
data/Blackstad_CA1/<subject>/<YYYY-MM-DD>/
	├─ *.pos  (position)
	├─ *.set  (settings)
	├─ *._t64 or TT*.t64 / *.t (MClust cuts)
	└─ *.1..*.8 (tetrode files)
```

## CLI (optional)

```
python scripts/make_maps.py data/Blackstad_CA1/24116/2018-04-05 --bins 40 --sigma 1.0
```
