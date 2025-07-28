It looks like you've provided a comprehensive list of files and their respective descriptions for a project on **HO-3D dataset** for hand and object pose estimation.

To help you with the **Streamlit** application and other utilities you need for **HO-3D Dataset Visualizations and Evaluations**, here's a **README** file that includes the necessary details to set up and run the different scripts.

---

# **HO-3D Dataset Visualization and Evaluation**

This repository contains scripts to visualize and evaluate the **HO-3D** dataset used for **hand and object pose estimation**. The dataset includes various sequences with hand-object interaction. This project provides multiple utilities for processing and visualizing data as well as comparing different algorithms with manual annotations.

## **Key Features**

* **Visualization Tools**: Visualize hand-object interactions using Open3D and matplotlib.
* **Model Evaluation**: Evaluate the 3D joint and mesh predictions using Mean Squared Error (MSE) and F-Score metrics.
* **Data Preprocessing**: Convert the HO-3D dataset into usable formats, including mesh generation for hand and object models.
* **Pose Estimation**: Utilize the MANO model to estimate hand poses and visualize 3D meshes.
* **Camera Registration**: Visualize point clouds using multiple camera sequences.

## **Installation**

### **Dependencies**

Make sure you have the following libraries installed:

* **Open3D**: For 3D mesh visualization and point cloud manipulation.
* **NumPy**: For numerical operations.
* **Matplotlib**: For plotting images and visualizations.
* **Chumpy**: For optimization tasks (used with MANO).
* **CV2**: For image processing tasks.
* **Pickle**: For loading and saving model data.

Install the required libraries with:

```bash
pip install open3d numpy matplotlib chumpy opencv-python
```

### **MANO Model Setup**

* Download the **MANO** model and store it in the `./mano` directory.
* Follow the setup instructions in `setup_mano.py` to ensure the necessary MANO files are properly configured.

```bash
python setup_mano.py <path_to_original_mano_repo>
```

Ensure that the following files are present:

* `models/MANO_RIGHT.pkl`
* `webuser/smpl_handpca_wrapper_HAND_only.py`
* `webuser/verts.py`

## **Scripts Overview**

### **1. `vis_ho3d.py`** – Visualization of HO-3D Dataset

This script visualizes the projections of the **HO-3D dataset** using either **Open3D** or **Matplotlib**.

* **Arguments**:

  * `ho3d_path`: Path to the HO-3D dataset.
  * `ycbModels_path`: Path to the YCB model directory.
  * `split`: The dataset split (train/evaluation).
  * `seq`: The sequence name to visualize.
  * `id`: The image ID to visualize.
  * `visType`: Type of visualization (Open3D or Matplotlib).

**Run the script**:

```bash
python vis_ho3d.py --ho3d_path <HO3D dataset path> --ycbModels_path <YCB Models Path> --visType open3d
```

This will visualize a random sample from the dataset in **Open3D** or **Matplotlib**.

### **2. `compare_manual_anno.py`** – Comparison with Manual Annotations

This script compares the predicted hand joint locations with manually annotated ground truths and computes the **Mean Squared Error (MSE)**.

* **Arguments**:

  * `base_path`: Path to the HO-3D dataset.

**Run the script**:

```bash
python compare_manual_anno.py --base_path <HO3D dataset path>
```

This will calculate the MSE for each frame in the dataset and output the results.

### **3. `eval.py`** – Evaluate Pose Estimation Models

This script evaluates hand and object pose estimation models. It computes various metrics like **F-Score** and **Precision/Recall** at different thresholds for both hand joints and mesh vertices.

**Run the script**:

```bash
python eval.py --input_dir <input directory with ground truth and predictions> --output_dir <output directory for results>
```

This will evaluate the model predictions and output the results including **F-Score** metrics.

### **4. `pred.py`** – Prediction Script

This script is for generating predictions using your own hand and object pose estimation algorithm. The predictions will be saved in a JSON format.

**Run the script**:

```bash
python pred.py --base_path <HO3D dataset path> --out <output file path>
```

### **5. `prep_mesh.py`** – Generate Meshes for 3D Objects and Hands

This script generates **3D meshes** for hand and object models and saves them in **Wavefront OBJ** format using **Open3D**.

**Run the script**:

```bash
python prep_mesh.py --meta_data_path <HO3D dataset path> --ycbModels_path <YCB Models Path>
```

### **6. `setup_mano.py`** – Setup for MANO Model Files

This script sets up the necessary **MANO model files** by downloading or linking them and applies patches to ensure compatibility with the provided code.

**Run the script**:

```bash
python setup_mano.py <path_to_mano_model_repo>
```

### **7. `vis_pcl_all_cameras.py`** – Visualize Point Clouds from All Cameras

This script visualizes the **point clouds** generated from multiple cameras for each sequence in the dataset.

**Run the script**:

```bash
python vis_pcl_all_cameras.py --base_path <HO3D dataset path>
```

---

## **Usage Workflow**

1. **Prepare the Dataset**:

   * Ensure the HO-3D dataset is organized and the **MANO** model is set up properly.

2. **Run the Visualization**:

   * Use `vis_ho3d.py` to visualize a sample from the dataset.

3. **Compare Predictions**:

   * Use `compare_manual_anno.py` to compute the MSE between predicted and manual annotations.

4. **Evaluate Model**:

   * Use `eval.py` to evaluate your model's performance on the HO-3D dataset.

5. **Generate Meshes**:

   * Use `prep_mesh.py` to generate and export 3D meshes.

6. **Prediction**:

   * Use `pred.py` to generate predictions with your own pose estimation algorithm.

## **Results and Metrics**

The following metrics will be generated as part of the evaluation:

* **Mean Squared Error (MSE)**: Measures the error between predicted and ground truth joint locations.
* **F-Score**: Measures the performance of the model at different thresholds.
* **Point Cloud Visualizations**: Displays the 3D pose estimation results for hands and objects.

