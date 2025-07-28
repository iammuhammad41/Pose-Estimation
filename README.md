# HO-3D Dataset Hand and Object Pose Estimation

This project contains scripts for hand and object pose estimation using the **HO-3D** dataset. The dataset includes 3D annotations of hands and objects in various poses, captured using multi-camera setups. The provided scripts allow you to visualize, compare manual annotations, evaluate model predictions, and generate mesh data for 3D visualization.



## Table of Contents

1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [Usage](#usage)

   * [Visualizing the HO-3D dataset](#visualizing-the-ho-3d-dataset)
   * [Comparing Manual Annotations](#comparing-manual-annotations)
   * [Evaluation](#evaluation)
   * [Mesh Generation](#mesh-generation)
4. [Setup MANO Model](#setup-mano-model)
5. [File Descriptions](#file-descriptions)


## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your_username/ho3d-pose-estimation.git
   cd ho3d-pose-estimation
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Follow the setup instructions to download the **MANO model** and other dependencies.


## Dependencies

* Python 3.7+
* Open3D
* NumPy
* Matplotlib
* OpenCV
* Chumpy
* tqdm
* json
* argparse

For the complete list of dependencies, refer to the `requirements.txt` file.


## Usage

### Visualizing the HO-3D dataset

To visualize the projections in the HO-3D dataset, run the `visualize_ho3d.py` script. This will show you either **Open3D** or **Matplotlib** visualizations of hand and object poses.

```bash
python visualize_ho3d.py --ho3d_path <path_to_ho3d_dataset> --ycbModels_path <path_to_ycb_models> --split <train/evaluation> --seq <sequence_name> --id <image_id> --visType <open3d/matplotlib>
```

* `ho3d_path`: Path to the HO3D dataset directory.
* `ycbModels_path`: Path to the YCB models directory.
* `split`: Split type (`train` or `evaluation`).
* `seq`: Sequence name to visualize (optional).
* `id`: Image ID to visualize (optional).
* `visType`: Type of visualization (`open3d` or `matplotlib`).


### Comparing Manual Annotations

To compare model predictions with manual annotations, use the `compare_manual_annotations.py` script.

```bash
python compare_manual_annotations.py --base_path <path_to_ho3d_dataset>
```

This will compute the **Mean Squared Error (MSE)** for hand joint locations based on manual annotations.


### Evaluation

To evaluate model predictions, use the `eval_metrics.py` script. It computes **F-scores**, **precision**, **recall**, and other metrics for 3D keypoints and meshes.

```bash
python eval_metrics.py --gt_path <path_to_ground_truth> --pred_path <path_to_predictions> --output_dir <output_directory>
```

This will generate evaluation metrics such as the average 3D keypoint error and the F-score at various thresholds.

### Mesh Generation

To generate 3D mesh data from the HO-3D dataset, use the `prep_mesh.py` script. It converts 3D hand and object meshes into a format compatible with Open3D.

```bash
python prep_mesh.py --meta_data_path <path_to_ho3d_dataset> --ycbModels_path <path_to_ycb_models>
```

This script will generate mesh files (`.obj`) for hand and object models and save them to the dataset directories.


## Setup MANO Model

Before using the `mano` model for 3D pose estimation, you need to set it up by running the following script:

```bash
python setup_mano.py --mano_path <path_to_mano_repository>
```

This script sets up the MANO model required for generating hand meshes and joint poses.


## File Descriptions

* **`visualize_ho3d.py`**: Visualizes 3D projections of hand and object poses from the HO-3D dataset.
* **`compare_manual_annotations.py`**: Compares model predictions with manual annotations and calculates the Mean Squared Error (MSE).
* **`eval_metrics.py`**: Computes evaluation metrics such as F-score, precision, recall, and 3D keypoint errors.
* **`prep_mesh.py`**: Generates 3D mesh data from the HO-3D dataset.
* **`setup_mano.py`**: Sets up the MANO model and patches required files for hand pose estimation.


### Notes:

* Make sure to download the **MANO model** and **YCB models** for proper functioning.
* You may need to adjust paths in the script to point to the correct directories where your dataset and models are stored.
* For further details, visit: (https://github.com/zerchen/ho3d), (https://github.com/shreyashampali/ho3d)
### Acknowledgment

1. https://github.com/zerchen/ho3d
2. https://github.com/shreyashampali/ho3d)
