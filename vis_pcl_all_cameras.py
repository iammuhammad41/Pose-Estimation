import open3d as o3d
import os
import cv2
import argparse
import numpy as np

height, width = 480, 640
depth_threshold = 800
multi_cam_seqs = ['ABF1', 'BB1', 'GPMF1', 'GSF1', 'MDF1', 'SB1']

def load_point_clouds(seq, fID, base_path):
    """
    Load point clouds from a specific sequence and file ID.
    """
    pcds = []
    seq_dir = os.path.join(base_path, seq)
    calib_dir = os.path.join(base_path, '..', 'calibration', seq, 'calibration')
    cams_order = np.loadtxt(os.path.join(calib_dir, 'cam_orders.txt')).astype('uint8').tolist()

    for cam in cams_order:
        color_raw = o3d.io.read_image(os.path.join(seq_dir, f'{cam}', 'rgb', f'{fID}.png'))
        depth_raw = cv2.imread(os.path.join(seq_dir, f'{cam}', 'depth', f'{fID}.png'))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(width, height))
        pcds.append(pcd)

    return pcds


def combine_point_clouds(pcds):
    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd
    return pcd_combined


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])


def main(base_path):
    for seq in multi_cam_seqs:
        files = os.listdir(os.path.join(base_path, seq, 'rgb'))
        for file_id in files:
            pcds = load_point_clouds(seq, file_id, base_path)
            combined_pcd = combine_point_clouds(pcds)
            visualize_point_cloud(combined_pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str, help='Path to the HO-3D dataset')
    args = parser.parse_args()
    main(args.base_path)
