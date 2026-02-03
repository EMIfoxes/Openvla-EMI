import os.path

# ruff: noqa: E402
# 这是一个注释，用于禁用ruff（一个Python代码格式化工具）的E402错误检查。
# E402通常表示模块导入位置不符合PEP 8规范（PEP 8建议所有导入语句应该放在文件的开头）。
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback #导入traceback模块，用于捕获和打印异常的堆栈信息。
from dataclasses import dataclass #dataclass用于简化类的定义，特别是那些主要用于存储数据的类。
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn #uvicorn是一个ASGI（Asynchronous Server Gateway Interface）服务器，用于运行异步Web应用程序。
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import get_vla,get_vla_action,get_action_head,get_processor,get_proprio_projector
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        这是一个文档字符串，用于描述类的功能。它说明了 OpenVLAServer 是一个简单的服务器，用于 OpenVLA 模型，提供 /act 接口以根据输入的观察和指令预测动作。
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg) #根据配置加载 OpenVLA 模型

        # Load proprio projector
        self.proprio_projector = None 
        # 如果配置中 use_proprio 为 True，则加载本体感知投影器（proprio_projector）。
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            action = get_vla_action(
                self.cfg, self.vla, self.processor, observation, instruction, action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=self.cfg.use_film,
            )

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration  服务器配置
    host: str = "0.0.0.0"            # Host IP Address 服务器的IP地址，默认为 "0.0.0.0"，表示服务器可以监听所有网络接口上的连接。
    port: int = 8777                 # Host Port 服务器的端口号，默认为 8777

    #################################################################################################################
    # Model-specific parameters  模型特定参数
    #################################################################################################################
    model_family: str = "openvla"                    # 模型家族名称，默认为 "openvla" Model family 
    pretrained_checkpoint: Union[str, Path] = ""     # 训练模型的检查点路径，可以是字符串或 Path 对象。 Pretrained checkpoint path 

    use_l1_regression: bool = True                   # 如果为 True，则使用 L1 回归目标的连续动作头。 If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # 如果为 True，则使用扩散建模目标（DDIM）的连续动作头。 If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # 如果为 True，则使用 FiLM（Feature-wise Linear Modulation）将语言输入注入到视觉特征中。 If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # VLA 输入中的图像数量，默认为 3。 Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # 是否在输入中包含本体感知状态。 Whether to include proprio state in input

    center_crop: bool = True                         # 是否进行中心裁剪（如果训练时使用了随机裁剪的图像增强）。 Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # 权重矩阵的秩，需要与训练时的设置一致。 Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # 动作反归一化的键。 Action un-normalization key
    use_relative_actions: bool = False               # 是否使用相对动作（关节角度的变化量）。 Whether to use relative actions (delta joint angles)
    # 是否使用 8 位或 4 位量化加载模型（仅适用于 OpenVLA）。
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # 随机种子，用于确保结果的可重复性。 Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
