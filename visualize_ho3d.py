"""
Visualize projections in the custom HO-3D dataset for hand and object pose estimation
"""
import argparse
import os
import random
import cv2
import numpy as np
import open3d
from utils.visual_utils import *
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from mano import load_mano_model


MANO_MODEL_PATH = './models/MANO_RIGHT.pkl'

# mapping of joints from the MANO model order to simple order (thumb to pinky)
joints_map_mano_to_simple = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception("MANO model missing! Please run setup_mano.py to set up the MANO folder")
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model


def forward_kinematics(fullpose, translation, beta):
    """
    Convert MANO parameters to 3D points and mesh.
    :param fullpose: Hand pose parameters
    :param translation: Translation vector
    :param beta: Shape parameters
    :return: 3D joint positions and mesh
    """
    assert fullpose.shape == (48,)
    assert translation.shape == (3,)
    assert beta.shape == (10,)

    mano_model = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    mano_model.fullpose[:] = fullpose
    mano_model.trans[:] = translation
    mano_model.betas[:] = beta

    return mano_model.J_transformed.r, mano_model


def visualize_ho3d_dataset(ho3d_path, ycb_models_path, split, seq, image_id, visualization_type):
    """
    Visualize the 3D hand and object pose from the HO-3D dataset.
    :param ho3d_path: Path to the HO3D dataset
    :param ycb_models_path: Path to YCB object models
    :param split: Dataset split (train or evaluation)
    :param seq: Sequence name
    :param image_id: Image ID to visualize
    :param visualization_type: Type of visualization (open3d or matplotlib)
    """
    base_dir = ho3d_path
    ycb_models_dir = ycb_models_path
    split_type = split

    if seq is None:
        seq = random.choice(os.listdir(os.path.join(base_dir, split_type)))
        run_loop = True
    else:
        run_loop = False

    if image_id is None:
        image_id = random.choice(os.listdir(os.path.join(base_dir, split_type, seq, 'rgb'))).split('.')[0]

    if visualization_type == 'matplotlib':
        o3d_win = Open3DWin()

    while True:
        image_path = read_rgb_image(base_dir, seq, image_id, split_type)
        depth_path = read_depth_image(base_dir, seq, image_id, split_type)
        annotation = read_annotation(base_dir, seq, image_id, split_type)

        if annotation['objRot'] is None:
            print(f'Frame {image_id} in sequence {seq} does not have annotations')
            if not run_loop:
                break
            else:
                seq = random.choice(os.listdir(os.path.join(base_dir, split_type)))
                image_id = random.choice(os.listdir(os.path.join(base_dir, split_type, seq, 'rgb'))).split('.')[0]
                continue

        obj_corners = annotation['objCorners3DRest']
        obj_corners_trans = np.matmul(obj_corners, cv2.Rodrigues(annotation['objRot'])[0].T) + annotation['objTrans']

        if split_type == 'train':
            hand_joints_3d, hand_mesh = forward_kinematics(annotation['handPose'], annotation['handTrans'], annotation['handBeta'])

        if split_type == 'train':
            hand_kps = project_3d_points(annotation['camMat'], hand_joints_3d, is_opengl_coords=True)
        else:
            hand_kps = project_3d_points(annotation['camMat'], np.expand_dims(annotation['handJoints3D'], 0), is_opengl_coords=True)

        obj_kps = project_3d_points(annotation['camMat'], obj_corners_trans, is_opengl_coords=True)

        if visualization_type == 'open3d':
            obj_mesh = read_obj(os.path.join(ycb_models_dir, 'models', annotation['objName'], 'textured_simple.obj'))
            obj_mesh.v = np.matmul(obj_mesh.v, cv2.Rodrigues(annotation['objRot'])[0].T) + annotation['objTrans']
            open3d_visualize([hand_mesh, obj_mesh], ['r', 'g'])
        elif visualization_type == 'matplotlib':
            img_anno = show_hand_joints(image_path, hand_kps)
            img_anno = show_obj_joints(img_anno, obj_kps)
            plt.imshow(img_anno)
            plt.show()

        if run_loop:
            seq = random.choice(os.listdir(os.path.join(base_dir, split_type)))
            image_id = random.choice(os.listdir(os.path.join(base_dir, split_type, seq, 'rgb'))).split('.')[0]
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ho3d_path", type=str, help="Path to HO3D dataset")
    parser.add_argument("ycbModels_path", type=str, help="Path to YCB models directory")
    parser.add_argument("-split", required=False, type=str, choices=['train', 'evaluation'], default='train')
    parser.add_argument("-seq", required=False, type=str)
    parser.add_argument("-id", required=False, type=str)
    parser.add_argument("-visType", required=False, choices=['open3d', 'matplotlib'], default='matplotlib')
    args = parser.parse_args()

    visualize_ho3d_dataset(args.ho3d_path, args.ycbModels_path, args.split, args.seq, args.id, args.visType)
