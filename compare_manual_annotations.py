import os
import argparse
import numpy as np
from utils.visual_utils import load_pickle_data

finger_tip_ids = [20, 19, 18, 17, 16]

def calculate_mse_per_frame(manual_annotation, annotation_file):
    seq = annotation_file.split('_')[0]
    file_id = annotation_file.split('_')[1]

    output_dir = os.path.join(args.base_path, 'train', f'{seq}1', 'meta', f'{file_id}.pkl')

    if not os.path.exists(output_dir):
        print(f'[INFO] Skipping sequence {seq} file ID {file_id} as it is part of test set')
        return np.nan

    output_data = load_pickle_data(output_dir)
    hand_joint_locs = output_data['handJoints3D'][finger_tip_ids]

    mse = np.mean(np.linalg.norm(manual_annotation - hand_joint_locs, axis=1))
    return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare with manual annotations')
    parser.add_argument('base_path', type=str, help='Path to HO3D dataset')
    args = parser.parse_args()

    annot_save_dir = os.path.join(args.base_path, 'manual_annotations')
    annot_files = sorted(os.listdir(annot_save_dir))

    mse_sum = []
    for annot_file in annot_files:
        annot_data = np.load(os.path.join(annot_save_dir, annot_file))
        mse_sum.append(calculate_mse_per_frame(annot_data, annot_file[:-4]))

    mse_sum = np.array(mse_sum, dtype=np.float32)

    print(f'Number of samples = {mse_sum.shape[0] - np.sum(np.isnan(mse_sum))}')
    print(f'Average MSE = {np.nanmean(mse_sum)*1000} mm, Standard Deviation = {np.std(mse_sum[np.logical_not(np.isnan(mse_sum))])*1000} mm')
