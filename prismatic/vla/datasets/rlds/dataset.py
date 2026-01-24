"""
dataset.py
用于配置和初始化 RLDS 数据集的核心接口脚本。
Core interface script for configuring and initializing RLDS datasets.
"""

import copy
import inspect
import json
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import obs_transforms, traj_transforms
from prismatic.vla.datasets.rlds.utils import goal_relabeling, task_augmentation
from prismatic.vla.datasets.rlds.utils.data_utils import allocate_threads, get_dataset_statistics, normalize_action_and_proprio, pprint_data_mixture, tree_map

# 初始化监视器（Overwatch）=>> 封装了 logging.Logger Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# 配置 TensorFlow 以使用*无 GPU 设备*（以防止与 PyTorch 冲突）。 Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")


# ruff: noqa: B006
def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: ACTION_PROPRIO_NORMALIZATION_TYPE,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """

    此函数负责从存储中加载特定的 RLDS 数据集，并将其转换为标准化格式。生成一个包含轨迹的数据集。不包括 CPU 密集型操作。

    如果提供了 `standardize_fn`，则会将其应用于每个轨迹。此函数应将轨迹转换为标准格式，其中包括 "observation" 和 "action" 两个键。"observation" 键应是一个字典，包含一些额外的键，这些键将根据 `*_obs_keys` 参数提取为更标准的格式。

    `image_obs_keys` 和 `depth_obs_keys` 参数是从新名称到旧名称的映射，或者在旧名称的位置用 None 替换以插入填充。例如，如果在 `standardize_fn` 处理后，"observation" 字典中包含名为 "workspace" 和 "wrist" 的 RGB 图像，并且 `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`，那么生成的数据集将包含一个 "observation" 字典，其中包含 "image_primary"、"image_secondary" 和 "image_wrist" 这些键，其中 "image_primary" 对应于 "workspace"，"image_secondary" 是一个填充图像，而 "image_wrist" 对应于 "wrist"。

    `state_obs_keys` 是一个一维本体感知键的列表，这些键将被提取自 "observation" 字典，拼接成一个单一数组，并映射到 "observation" 字典中的 "proprio" 键。对于每个 None 条目，将插入一个单一的填充元素（零）。

    数据集还将包含一个 "task" 字典。如果提供了 `language_key`，则 "task" 字典将包含 "language_instruction" 键，从 `traj[language_key]` 中提取。

    参数：
        name (str): RLDS 数据集的名称（通常是 "name" 或 "name:version"）。
        data_dir (str): 数据目录的路径。
        train (bool): 是否使用训练集或验证集。
        shuffle (bool, 可选): 是否打乱文件读取顺序（不会完全打乱数据集，因为一个文件通常包含许多轨迹）！
        standardize_fn (Callable[[dict], dict], 可选): 如果提供，这将是应用于每个轨迹的第一个函数。
        image_obs_keys (Mapping[str, str|None]): 从 {新名称: 旧名称} 的映射，表示从 "observation" 字典中提取哪些 RGB 图像。
            `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`。
            如果 `old` 的值为 None，则插入一个填充图像（空字符串）。
        depth_obs_keys (Mapping[str, str|None]): 与 `image_obs_keys` 相同，但用于深度图像。键将以前缀 "depth_" 而不是 "image_" 开始。
        state_obs_keys (Sequence[str|None]): 从 "observation" 字典中提取的一维本体感知键的列表，拼接后映射到 "proprio"。
            对于每个 None 条目，插入一个单一的填充元素。
        language_key (str, 可选): 如果提供，"task" 字典将包含 "language_instruction" 键，从 `traj[language_key]` 中提取。
        action_proprio_normalization_type (str, 可选): 对动作、本体感知或两者执行的归一化类型。
            可以是 "normal"（均值为 0，标准差为 1）或 "bounds"（归一化到 [-1, 1]）。
        dataset_statistics (dict|str, 可选): 包含数据集统计信息的字典（或 JSON 文件路径），用于归一化。
            如果 `action_proprio_normalization_type` 是 "normal"，则应包含 "mean" 和 "std" 键。
            如果是 "bounds"，则应包含 "min" 和 "max" 键。还可以提供 "num_transitions" 和 "num_trajectories" 键以供下游使用（例如，在 `make_interleaved_dataset` 中）。
            如果未提供，则会即时计算统计信息。
        absolute_action_mask (Sequence[bool], 可选): 默认情况下，所有动作维度都被认为是相对的。
            这在 `future_action_window_size > 0` 时很重要：从轨迹末尾（或使用目标重标记时的目标时间步）之外采取的动作需要被设置为“中性”，以表示任务已完成。
            对于相对动作，“中性”意味着零，但对于绝对动作，“中性”意味着重复最后一个有效动作。如果提供此掩码，则指示哪些动作维度是绝对的。
        action_normalization_mask (Sequence[bool], 可选): 如果提供，指示哪些动作维度应被归一化。
            例如，如果某个动作维度始终恰好为 0 或 1，则可能不想对其进行归一化。默认情况下，所有动作维度都被归一化。
        num_parallel_reads (int): 并行读取工作线程的数量。默认为 AUTOTUNE。
        num_parallel_calls (int): 并行调用的数量，用于轨迹映射操作。默认为 AUTOTUNE。

    返回值：
        包含轨迹的数据集，其中每个步骤具有以下字段：
        - observation:
            - image_{name1, name2, ...} # RGB 图像观测
            - depth_{name1, name2, ...} # 深度图像观测
            - proprio                   # 一维本体感知观测数组
            - timestep                  # 每帧的时间步
        - task:
            - language_instruction      # 语言指令，如果提供了 `language_key` 则存在
        - action                        # 动作向量
        - dataset_name                  # 数据集名称

    This function is responsible for loading a specific RLDS dataset from storage and getting it into a standardized
    format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the trajectory
    into a standard format, which includes the keys "observation" and "action". Entry "observation" should be a
    dictionary containing some number of additional keys, which will be extracted into an even more standardized format
    according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in place of an
    old name to insert padding. For example, if after `standardize_fn`, your "observation" dict has RGB images called
    "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`, then
    the resulting dataset will have an "observation" dict containing the keys "image_primary", "image_secondary", and
    "image_wrist", where "image_primary" corresponds to "workspace", "image_secondary" is a padding image, and
    "image_wrist" corresponds to "wrist".

    Entry `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which will
    be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be inserted for each
    None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will contain the
    key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset, since one
            file usually contains many trajectories)!
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to extract from the
            "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`.
            If a value of `old` is None, inserts a padding image instead (empty string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from the
            "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding for each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key "language_instruction",
            extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)

    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of "relabeling"
    (e.g., filtering, chunking, adding goals, dropping keys).

    Transforms in this function should have the following properties:
        - They require access to an entire trajectory (i.e., they cannot be applied frame-wise).
        - They are generally not CPU-intensive, mostly involving moving and copying data.
        - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        future_action_window_size (int, optional): The number of future actions beyond window_size to include
            in the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")

        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action))

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["observation"]["proprio"]) <= max_proprio))

    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions, giving them a new axis at index 1 of size `window_size` and
    # `window_size + future_action_window_size`, respectively
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset


def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a frame level. These transforms are usually more CPU-intensive, (e.g.,
    decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # Convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame["task"] = fn(frame["task"])
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # Decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(obs_transforms.decode_and_resize, resize_size=resize_size, depth_resize_size=depth_resize_size),
        ),
        num_parallel_calls,
    )

    if train:
        # Augment all images with the same seed, skipping padding images
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    return dataset, dataset_statistics["num_trajectories"], dataset_statistics


# === 核心初始化器 Core Initializer === 创建交错数据集
def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    # rlds_config = dict(
    # traj_transform_kwargs=dict(
    #     window_size=1,                                      # 如果我们想要输入/预测多于一步的信息If we wanted to feed / predict more than one step
    #     future_action_window_size=NUM_ACTIONS_CHUNK-1,      # 用于动作分块 For action chunking
    #     skip_unlabeled=True,                                # 跳过没有语言标签的轨迹 Skip trajectories without language labels
    #     goal_relabeling_strategy="uniform",                 # 目前目标未被使用 Goals are currently unused
    # ),
    # frame_transform_kwargs=dict(
    #     resize_size=resize_resolution,
    #     num_parallel_calls=16,                              # 用于 CPU 密集型操作（解码、调整大小等） For CPU-intensive ops (decoding, resizing, etc.)
    # ),
    # dataset_kwargs_list=per_dataset_kwargs,
    # shuffle_buffer_size=shuffle_buffer_size,
    # sample_weights=weights,
    # balance_weights=True,
    # traj_transform_threads=len(mixture_spec),
    # traj_read_threads=len(mixture_spec),
    # train=train,)
    """

    从数据集配置列表(kwargs)创建一个交错数据集。返回一个批处理帧的数据集。

    Args:
        dataset_kwargs_list: kwargs 列表，每个元素都传递给 `make_dataset_from_rlds`。
            其中的 "num_parallel_calls" 和 "num_parallel_reads" 分别使用 `traj_transform_threads` 和`traj_read_threads` 覆盖。
        sample_weights: 列表中每个数据集的采样。权重如果为 None,则默认为均匀分布。
        train: 是否为训练数据集或验证数据集。
        shuffle_buffer_size: 数据集洗牌缓冲区的大小（以帧数计）。
        traj_transform_kwargs: 传递给 `apply_trajectory_transforms` 的 kwargs。
            其中的 "num_parallel_calls" 使用 `traj_transform_threads` 覆盖。
        frame_transform_kwargs: 传递给 `apply_frame_transforms` 的 kwargs。
        batch_size: 批量大小。如果未提供，则输出不进行批处理。
        balance_weights: 如果为 True,则采样权重会乘以每个数据集中的帧数。
            这样，如果所有采样权重相等，那么对交错数据集进行一次完整迭代将对应于对每个单独数据集进行一次完整迭代
            （仅在期望上成立，因为实际上采样是随机的）。
        traj_transform_threads: 轨迹变换的并行调用总数，根据采样权重在各数据集之间分配。
            如果为 None,则每个数据集默认为 AUTOTUNE。
        traj_read_threads: 轨迹变换的并行读取工作线程总数，根据采样权重在各数据集之间分配。
            如果为 None,则每个数据集默认为 AUTOTUNE。

    Creates an interleaved dataset from list of dataset configs (kwargs). Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overridden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overridden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # 如果未指定 `sample_weights`，则默认采用均匀采样。 Default to uniform sampling (if `sample_weights` is not specified)
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

    # 检查有效的 `traj_transform_kwargs` 和 `frame_transform_kwargs`。 Check valid `traj_transform_kwargs` and `frame_transform_kwargs`
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    # 获取数据集大小 Get Dataset Sizes
    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        _, dataset_statistics = make_dataset_from_rlds(**data_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # 获取“主要”数据集（即采样权重为 1.0 的数据集）的索引。 Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])

    # 平衡和归一化权重 Balance and Normalize Weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
    #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight == 1.0)
    # 有效数据集长度 = 每个数据集完成至少一个周期所需的样本数量
    #   =>> 注意 :: 仅计算“主要”数据集（即采样权重为 1.0 的数据集）
    dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

    # 根据权重分配线程 Allocate Threads based on Weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info("Threads per Dataset: %s", threads_per_dataset)
    overwatch.info("Reads per Dataset: %s", reads_per_dataset)

    # 构建数据集 Construct Datasets
    overwatch.info("Constructing datasets...")
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs")
            if "dataset_frame_transform_kwargs" in dataset_kwargs
            else {}
        )
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    # 在帧级别交错 Interleave at the Frame Level
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # 验证集 =>> 固定一个数据的洗牌缓冲区并将其缓存在 RAM 中；防止内存逐渐增加！
    # Validation =>> fix a single shuffle buffer of data and cache it in RAM; prevents gradual memory increase!
    if not train:
        dataset = dataset.take(shuffle_buffer_size).cache()

    # Shuffle the Dataset
    #   =>> IMPORTANT :: Shuffle AFTER .cache(), or else memory will still leak!
    # 洗牌数据集
    #   =>> 重要提示 :: 必须在调用 .cache() 之后进行洗牌，否则内存仍然会泄漏！
    dataset = dataset.shuffle(shuffle_buffer_size)

    # Apply Frame Transforms
    overwatch.info("Applying frame transforms on dataset...")
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # [Contract] When training VLA Policies, we let the Collator handle Batching!
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Note =>> Seems to reduce memory usage without affecting speed?
    dataset = dataset.with_ram_budget(1)

    # Save for Later
    dataset.sample_weights = sample_weights

    return dataset, dataset_len, all_dataset_statistics
