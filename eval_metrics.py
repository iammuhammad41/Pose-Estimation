import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from utils.visual_utils import *


def main(gt_path, pred_path, output_dir, pred_file_name='pred.json', set_name='evaluation'):
    """
    Evaluation loop to compute metrics such as MSE, precision, recall, and F-score.
    """
    assert os.path.exists(pred_path), f"Prediction directory '{pred_path}' does not exist"
    pred_file = os.path.join(pred_path, pred_file_name)

    # Load ground truth and prediction data
    xyz_list, verts_list = json_load(os.path.join(gt_path, f'{set_name}_xyz.json')), json_load(os.path.join(gt_path, f'{set_name}_verts.json'))
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Initialize evaluation utilities
    eval_xyz = EvalUtil()
    eval_mesh_err = EvalUtil(num_kp=778)
    f_score, f_score_aligned = [], []

    for idx in tqdm(range(len(xyz_list))):
        xyz_gt, verts_gt = np.array(xyz_list[idx]), np.array(verts_list[idx])
        xyz_pred, verts_pred = np.array(pred_data[0][idx]), np.array(pred_data[1][idx])

        # Evaluate F-scores
        f, _, _ = calculate_fscore(verts_gt, verts_pred, 0.01)
        f_score.append(f)

        # Alignment and scale correction
        aligned_pred = align_w_scale(xyz_gt, xyz_pred)
        eval_xyz.feed(xyz_gt, np.ones_like(xyz_gt[:, 0]), aligned_pred)

        # Record aligned errors
        eval_mesh_err.feed(verts_gt, np.ones_like(verts_gt[:, 0]), verts_pred)

    # Calculate results
    eval_results = eval_xyz.get_measures()
    print('Evaluation results:')
    print(f'Mean XYZ Error = {eval_results[0]:.3f} cm')
    print(f'F-score = {np.mean(f_score):.3f}')

    # Save results to file
    with open(os.path.join(output_dir, 'eval_results.txt'), 'w') as f:
        f.write(f'Mean XYZ Error = {eval_results[0]:.3f} cm\n')
        f.write(f'F-score = {np.mean(f_score):.3f}\n')

    print('Evaluation complete.')
