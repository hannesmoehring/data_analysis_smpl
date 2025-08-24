import os

import numpy as np

PATH_NJV = "datasets/humanml3d/new_joint_vecs"
PATH_NJ = "datasets/humanml3d/new_joints"


def check_data_consistency():
    max_changes: dict[str, float] = {}
    min_changes: dict[str, float] = {}
    for filename in os.listdir(PATH_NJV):
        njv = np.load(os.path.join(PATH_NJV, filename))
        nj = np.load(os.path.join(PATH_NJ, filename))
        # print("nj", len(nj.shape))
        # print("njv", len(njv.shape))
        if len(njv.shape) != 2 or len(nj.shape) != 3:
            continue
        if njv.shape[0] != nj.shape[0]:
            print(f"File {filename} has different first frame in njv and nj!")
            print("njv:", njv.shape[0])
            print("nj :", nj.shape[0])
            continue

        root = nj[:, 0:1, :]  # (T,1,3)
        rel_xyz = nj[:, 1:, :] - root  # (T,21,3)
        rel_changes = []
        for t in range(0, nj.shape[0]):
            parts = split_h3d_frame(njv[t])
            rel_pos = np.mean((rel_xyz[t] - parts["ric"]) ** 2)
            rel_changes.append(rel_pos)
            # print("MSE rel positions vs ric:", rel_changes)
        max_val: float = np.max(rel_changes)
        min_val: float = np.min(rel_changes)
        max_changes.update({filename: max_val})
        min_changes.update({filename: min_val})

    max_values = list(max_changes.values())
    min_values = list(min_changes.values())
    print(max_changes)
    print(min_changes)
    print(f"Maximum total change: {np.max(max_values)} at {list(max_changes.keys())[np.argmax(max_values)]}")
    print(f"Maximum negative change: {np.min(min_values)} at {list(min_changes.keys())[np.argmin(min_values)]}")


def split_h3d_frame(frame_vec, num_joints=22):
    i = 0
    root_rot_vel = frame_vec[i]
    i += 1  # (1,)
    root_lin_vel = frame_vec[i : i + 2]
    i += 2  # (2,)  vx, vz
    root_y = frame_vec[i]
    i += 1  # (1,)

    ric_len = (num_joints - 1) * 3
    rot6_len = (num_joints - 1) * 6
    lvel_len = num_joints * 3

    ric = frame_vec[i : i + ric_len].reshape(num_joints - 1, 3)
    i += ric_len
    rot6 = frame_vec[i : i + rot6_len].reshape(num_joints - 1, 6)
    i += rot6_len
    local_vel = frame_vec[i : i + lvel_len].reshape(num_joints, 3)
    i += lvel_len
    foot = frame_vec[i : i + 4]

    return {
        "root_rot_vel": root_rot_vel,
        "root_lin_vel": root_lin_vel,
        "root_y": root_y,
        "ric": ric,  # joints 1..21 (relative to root & root frame)
        "rot6d": rot6,  # joints 1..21
        "local_vel": local_vel,  # joints 0..21
        "foot_contact": foot,
    }
