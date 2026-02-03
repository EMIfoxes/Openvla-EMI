"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
`run_libero_eval.py`

在 LIBERO 仿真基准任务套件中评估一个已训练的策略。
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

import argparse
import cv2
import datetime
import h5py # 用于读写HDF5文件，HDF5是一种用于存储大量数据的文件格式，常用于深度学习和科学计算中
# import init_path
import json
import numpy as np
import os
import time
from glob import glob

import robosuite as suite
from robosuite import load_controller_config # 用于加载控制器的配置
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action # 将用户输入转换为机器人动作。

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from libero.libero import get_libero_path

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import get_libero_dummy_action,get_libero_image,get_libero_wrist_image,quat2axisangle,save_rollout_video
from experiments.robot.openvla_utils import get_action_head,get_noisy_action_projector,get_processor,get_proprio_projector,resize_image_for_policy
from experiments.robot.robot_utils import DATE_TIME,get_action,get_image_resize_size,get_model,invert_gripper_action,normalize_gripper_action,set_seed_everywhere
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def collect_human_trajectory(env, device, arm, env_configuration, problem_info, remove_directory=[]):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration

    使用设备(键盘或SpaceNav 3D鼠标)来收集演示。
    回放轨迹以npz格式保存到文件中。
    修改DataCollectionWrapper包装器以添加新字段或更改数据格式。

    参数：
        env (MujocoEnv): 要控制的环境
        device (Device): 从设备接收控制
        arms (str): 要控制哪只手臂（例如双臂操作）'right' 或 'left'
        env_configuration (str): 指定的环境配置
    """

    # 初始化和环境重置
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 总是对应于 agentview ID = 2 always corresponds to agentview
    env.render() # 渲染环境
    
    task_completion_hold_count = (-1) 
        
    # 用于在达到目标后收集10个时间步的计数器 counter to collect 10 timesteps after reaching goal
    device.start_control()

    # 循环直到我们从输入中接收到重置信号或任务完成 Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    while True:
        count += 1
        # Set active robot
        active_robot = (env.robots[0]if env_configuration == "bimanual" else env.robots[arm == "left"])
            
        # 获取最新的动作！！！ Get the newest action
        action, grasp = input2action(device=device,robot=active_robot,active_arm=arm,env_configuration=env_configuration)
            
        # 如果动作为空，则表示这是一个重置信号，因此我们应该退出循环 If action is none, then this a reset so we should break
        if action is None:
            print("Break")
            saving = False
            break

        # 运行环境步骤 Run environment step

        env.step(action)
        env.render()
        # 如果任务完成，也退出循环 Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # 状态机用于检查是否连续10个时间步成功 state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # 锁定状态，递减计数 latched state, decrement count
            else:
                task_completion_hold_count = 10  # 在第一个成功的时间步重置计数 reset count on first success timestep
        else:
            task_completion_hold_count = -1  # 如果没有成功，则将计数器置为无效 null the counter if there's no success

    print(count)
    # 数据收集剧集结束时的清理操作 cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, args, remove_directory=[]):
    """
    将保存在 @directory 中的演示数据汇总到一个 hdf5 文件中。

    hdf5 文件的结构如下。

    data (组)
        date (属性) - 收集数据的日期
        time (属性) - 收集数据的时间
        repository_version (属性) - 收集数据时使用的代码库版本
        env (属性) - 收集演示数据的环境名称

        demo1 (组) - 每个演示数据都有一个组
        model_file (属性) - 演示数据的模型 XML 字符串
        states (数据集) - 展平的 Mujoco 状态
        actions (数据集) - 演示过程中应用的动作

        demo2 (组)
         ...

    参数：
        directory (str)：包含原始演示数据的目录路径。
        out_dir (str)：存储 hdf5 文件的路径。
        env_info (str)：包含环境信息的 JSON 编码字符串，包括控制器和机器人信息。

    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # GEN
    #################################################################################################################
    bddl_file:str = '/media/lxx/Elements/project/LIBERO/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl'
    controller:str = "OSC_POSE"
    robots:str = "Panda"
    config:str = "single-arm-opposed"
    camera:str = "agentview"
    directory:str = "demonstration_data"
    num_demonstration:int = 2
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # 任务套件名称 Task suite
    num_steps_wait: int = 10                         # 在仿真中等待物体稳定所需的步数 Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # 每个任务的试验次数 Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # “DEFAULT”，或初始状态 JSON 文件的路径 "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # 环境图像的分辨率（非策略输入分辨率）Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # 附加到运行 ID 末尾的额外说明，用于日志记录 Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # 本地目录，用于存储评估日志 Local directory for eval logs

    use_wandb: bool = False                          # 是否同时在 Weights & Biases 中记录结果 Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # WandB 实体的名称 Name of WandB entity
    wandb_project: str = "your-wandb-project"        # WandB 项目的名称 Name of WandB project

    seed: int = 7                                    # 随机种子（用于结果可复现性） Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "“期望 `center_crop==True`，因为模型是在使用图像增强的情况下训练的！” Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "不能同时用八位和四位量化 Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """初始化模型及其相关组件。Initialize model and associated components."""
    # 载入模型 Load model
    model = get_model(cfg)

    # 如果需要，加载本体感知投影器 Load proprio projector if needed 
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg,model.llm_dim,proprio_dim=8) # 8-dimensional proprio for LIBERO

    # 如果需要，加载动作头。 Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # 如果使用扩散模型，则加载噪声动作投影器。 Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # 如果需要，获取OpenVLA处理器。 Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """检查模型是否包含动作反归一化的键。 Check that the model contains the action un-normalization key."""
    # 初始化`unnorm_key` Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # 在某些情况下，该键必须手动修改（例如，在使用数据集的修改版本进行训练后，数据集名称中带有后缀 "_no_noops"）。
    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # 在`cfg`中设置`unnorm_key`。 Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """ 加载给定任务的初始状态。 Load initial states for the given task."""
    # 获取默认初始状态 Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # 如果使用自定义初始状态，则从文件中加载它们 If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """在环境中运行一个单独的剧集。 Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # 如果提供了初始状态，则设置初始状态（初始状态不是必须的） Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
        # obs, _, _, _ = env.step([0.0] * 7)

    # 初始化动作队列 Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def get_libero_env(task, model_family, resolution=256):
    """ 初始化并返回 LIBERO 环境以及任务描述。Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # 得到任务 Get task
    task = task_suite.get_task(task_id)

    # 得到初始状态 Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # 初始化环境和得到任务描述 Initialize environment and get task description
    # env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    #--------------------------------------------------------------------
    # 获取控制器配置 Get controller config
    controller_config = load_controller_config(default_controller=cfg.controller)

    # 创建参数配置 Create argument configuration
    config = {
        "robots": cfg.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(cfg.bddl_file)
    problem_info = BDDLUtils.get_problem_info(cfg.bddl_file)
    # 检查我们是否使用的是多臂机器人环境，如果是，则使用env_configuration参数。 Check if we're using a multi-armed environment and use env_configuration argument if so

    # 创建环境 Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = cfg.config
    print(language_instruction)

    env = TASK_MAPPING[problem_name](
        bddl_file_name=cfg.bddl_file,
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=cfg.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,)

    env_info = json.dumps(config)

    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),)
    
    env = DataCollectionWrapper(env, tmp_directory)

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(cfg.directory,f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"+ language_instruction.replace(" ", "_").strip('""'),)

    os.makedirs(new_dir)


    remove_directory = []
    i = 0
    while i < cfg.num_demonstration:
        print(i)
        # device改称model
        saving = collect_human_trajectory(env, device, args.arm, args.config, problem_info, remove_directory)
            
        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, args, remove_directory)
            i += 1


    # 开始剧集 Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # 处理初始状态 Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # 使用默认初始状态 Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # 获取从 JSON 文件中获取初始剧集状态的键 Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # 如果专家演示未能完成任务，则跳过该剧集。 Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file)
            


        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    """主函数，用于在 LIBERO 基准任务上评估已训练的策略。"""

    # 验证配置 Validate configuration
    validate_config(cfg)

    # 设计随机种子 Set random seed
    set_seed_everywhere(cfg.seed)

    # 初始化模型和组件 Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # 获取预期的图像尺寸 Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
        # 获取控制器配置 Get controller config

    controller_config = load_controller_config(default_controller=cfg.controller)

    # 创建参数配置 Create argument configuration
    config = {
        "robots": cfg.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(cfg.bddl_file)
    problem_info = BDDLUtils.get_problem_info(cfg.bddl_file)
    # 检查我们是否使用的是多臂机器人环境，如果是，则使用env_configuration参数。 Check if we're using a multi-armed environment and use env_configuration argument if so

    # 创建环境 Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = cfg.config
    print(language_instruction)


    env = TASK_MAPPING[problem_name](
        bddl_file_name=cfg.bddl_file,
        **config,
        has_renderer=False,           # True
        has_offscreen_renderer=False, # False
        render_camera=cfg.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,)



    env.reset()

    obs, _, _, _ = env.step([0.0] * 7)

    # 初始化动作队列 Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)



if __name__ == "__main__":
    eval_libero()
