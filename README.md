# 2D Laser People Benchmark - FROG Dataset

[[Preprint (arXiv)]](http://whatever.org) [[Dataset]](https://robotics.upo.es/datasets/frog/laser2d_people/)

## Requirements

- Python 3
- TensorFlow 2
- NumPy
- scikit-learn
- matplotlib
- tqdm
- h5py
- ROS 1 Noetic (only for bag conversion):
	- rospy
	- rosbag
	- geometry_msgs
	- nav_msgs
	- sensor_msgs

## Structure

- `/`: Contains the main scripts. Model weights (.h5) are read from here during evaluation.
	- `utils/`: Contains helper Python modules for FROG Benchmark.
	- `out/`: Contains subfolders, each corresponding to a training job. The training scripts write here final model weights (.h5), training metrics and diagrams.
	- `test/`: Contains evaluation files: inference results, calculated PR curves. (The folder must be manually created)

## Usage

This repository contains a collection of Python scripts, grouped in three main categories:

### Data processing

These scripts implement the conversion pipeline that generated the final dataset files. ROS bags are used as input, together with CSV files generated by [`laserscan_labeler`](https://github.com/robotics-upo/laserscan_labeler). The original ROS bags used to create the FROG dataset [can be found online](https://robotics.upo.es/datasets/frog/upo/).

- `convert_frog_bags.py`: This script extracts laser scan and odometry messages from ROS bags, as intermediate NumPy files.
- `convert_frog_circles.py`: This script converts CSV files generated by `laserscan_labeler` into intermediate NumPy files.
- `export_dataset.py`: This script combines the output of the previous scripts into finalized HDF5 files. It also optionally combines bags and generates a train/val split.

### Training and inference

These scripts implement the training and inference pipelines for our proposed models (LFE and PPN).

- `train_segmentation.py`: Main script used to train the LFE backbone, using the segmentation problem described in the paper.
- `train_localization.py`: Main script used to train LFE-PPN. This script can either train the backbone from scratch, or reuse a pre-trained LFE backbone.
- `test_segmentation.py`: Script used to evaluate LFE-Peaks, i.e. the LFE backbone used directly as a segmentation network, with a classic algorithm used to find peaks of the output signal. This script performs inference over the entire FROG test set, and stores the detections in a file (format detailed below).
- `test_localization.py`: Similar to above, this script implements inference for LFE-PPN.

### Benchmark evaluation

These scripts implement a common quantitative evaluation protocol on the FROG test set of 2D people detection models. As such, we intend future researchers to use them in order to provide standardized results and reduce variability.

- `convert_dr_spaam_output.py`: Script that converts the output of DR-SPAAM into the common inference results format used by this benchmark codebase.
- `calc_pr_curve.py`: Script that calculates the points of a model's PR curve using inference results.
- `benchmark.py`: Main benchmark script that takes as input the PR curves of different models, plots a diagram with them, and calculates the final metrics.

## Inference results

Inference results for all models are generated as and expected to be NumPy compressed files (.npz) containing the following arrays:

- `circles`: array of shape `(M, 3)` and type `np.float32` where M is the total number of detections across the entire dataset. Each row contains three columns: detection score (0.0~1.0), X and Y positions.
- `circle_idx` and `circle_num`: arrays of shape `(N,)` and type `np.uint32` where N is the number of scans in the test set. Similar to the FROG dataset format, these two arrays associate each scan in the test set with a slice of the `circles` array (containing detections).

## Configuration

Each provided script starts with a series of Python variable assignments which can be used to configure their behavior. Refer to each individual script's source code for more details.

## Reference

```
@article{frog2023,
	tbd = "TBD"
}
```