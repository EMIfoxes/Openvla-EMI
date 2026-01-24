"""
对某个套件任务中的单个任务进行转换，原本是一次转换一个套件中的十个任务，现在一次转换一个任务。

通过在环境中重放演示来重新生成LIBERO数据集(HDF5文件)。

注意事项：
    - 我们以256x256像素分辨率保存图像观测数据(而不是128x128)。
    - 我们过滤掉“无操作”（零）动作的转换，这些动作不会改变机器人的状态。
    - 我们过滤掉不成功的演示。
    - 在LIBERO HDF5数据转换为RLDS数据的过程中(此处未显示),我们将图像旋转180度,因为我们观察到环境返回的图像在我们的平台上是上下颠倒的。

用法：
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [libero_spatial | libero_object | libero_goal | libero_10] \
        --libero_raw_data_dir <原始HDF5数据集目录的路径> \
        --libero_target_dir <目标目录的路径>

示例(LIBERO-Spatial):
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite libero_spatial \
        --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
        --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""
import sys
sys.path.append('/media/lxx/Elements/project/openvla-oft/LIBERO')
import argparse
import json
import os
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action,get_libero_env

IMAGE_RESOLUTION = 256

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    返回一个动作是否为无操作(no-op)动作。

    无操作动作满足以下两个标准：
        (1) 除了最后一个维度（夹爪动作）之外，所有动作维度的值都接近于零。
        (2) 夹爪动作与前一个时间步的夹爪动作相同。

    关于 (2) 的解释：
        仅使用标准 (1) 来过滤动作是不够的，因为这样会移除机器人静止但正在打开或关闭夹爪的动作。
        因此，还需要考虑当前状态（通过检查前一个时间步的夹爪动作作为代理），以确定该动作是否真的为无操作。
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action

def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
        if user_input != 'y':
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    task_id = 0  # 需要对应更改任务 ！！！
    # Get task in suite
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

    # Get dataset for task
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]

    # Create new HDF5 file for regenerated demos
    new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
    new_data_file = h5py.File(new_data_path, "w")
    grp = new_data_file.create_group("data")

    for i in range(len(orig_data.keys())):
        # Get demo data
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        # Reset environment, set initial state, and wait a few steps for environment to settle
        env.reset()
        env.set_init_state(orig_states[0])
        for _ in range(10):
            obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

        # Set up new data lists
        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []

        # Replay original demo actions in environment and record observations
        for _, action in enumerate(orig_actions):
            # Skip transitions with no-op actions
            prev_action = actions[-1] if len(actions) > 0 else None
            if is_noop(action, prev_action):
                print(f"\tSkipping no-op action: {action}")
                num_noops += 1
                continue

            if states == []:
                # In the first timestep, since we're using the original initial state to initialize the environment,
                # copy the initial state (first state in episode) over from the original HDF5 to the new one
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                # For all other timesteps, get state from environment and record it
                states.append(env.sim.get_state().flatten())
                robot_states.append(np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]))

            # Record original action (from demo)
            actions.append(action)

            # Record data returned by environment
            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(np.hstack((obs["robot0_eef_pos"],T.quat2axisangle(obs["robot0_eef_quat"]),)))
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

            # Execute demo action in environment
            obs, reward, done, info = env.step(action.tolist())

        # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
        if done:
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            assert len(actions) == len(agentview_images)

            ep_data_grp = grp.create_group(f"demo_{i}")
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
            obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
            obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
            ep_data_grp.create_dataset("actions", data=actions)
            ep_data_grp.create_dataset("states", data=np.stack(states))
            ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
            ep_data_grp.create_dataset("rewards", data=rewards)
            ep_data_grp.create_dataset("dones", data=dones)

            num_success += 1

        num_replays += 1

        # Record success/false and initial environment state in metainfo dict
        task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{i}"
        if task_key not in metainfo_json_dict:
            metainfo_json_dict[task_key] = {}
        if episode_key not in metainfo_json_dict[task_key]:
            metainfo_json_dict[task_key][episode_key] = {}
        metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
        metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

        # Write metainfo dict to JSON file
        # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
        with open(metainfo_json_out_path, "w") as f:
            json.dump(metainfo_json_dict, f, indent=2)

        # Count total number of successful replays so far
        print(f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)")

        # Report total number of no-op actions filtered out so far
        print(f"  Total # no-op actions filtered out: {num_noops}")

    # Close HDF5 files
    orig_data_file.close()
    new_data_file.close()
    print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default='libero_goal',choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", )
    parser.add_argument("--libero_raw_data_dir", type=str,default='/media/lxx/Elements/project/openvla-oft/datasets/libero_goal',
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", )
    parser.add_argument("--libero_target_dir", type=str,default='/media/lxx/Elements/project/openvla-oft/datasets/libero_goal_no_noops',
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", )
    args = parser.parse_args()

    # Start data regeneration
    main(args)
