# OpenVLA-OFT in the LIBERO Simulation Benchmark

## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: OpenVLA-specific eval utils
  * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/finetune.py`: VLA fine-tuning script

**评估**

- **experiments/robot/libero/**：LIBERO评估文件  
    **run_libero_eval.py**：LIBERO评估脚本  
    **libero_utils.py**：LIBERO评估工具  
- **experiments/robot/**：通用评估工具文件  
    **openvla_utils.py**：OpenVLA专用评估工具  
    **robot_utils.py**：其他评估工具  

**训练**

- **vla-scripts/finetune.py**：VLA微调脚本

## Setup

Set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir
```

(Optional, if you plan to launch training) To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained OpenVLA-OFT checkpoints below.
Note that these are the same datasets used in the original OpenVLA project. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).

（可选，如果您计划启动训练）要下载我们在微调实验中使用的LIBERO数据集，请运行以下命令。这将下载LIBERO-Spatial、LIBERO-Object、LIBERO-Goal和LIBERO-10数据集，格式为RLDS数据（总计约10 GB）。您可以使用这些数据集来微调OpenVLA或训练其他方法。由于我们在下方提供了预训练的OpenVLA-OFT检查点，因此这一步是可选的。请注意，这些数据集与原始OpenVLA项目中使用的是相同的。如果需要，可以在此处查看如何下载原始非RLDS数据集的详细信息这里。
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## Launching LIBERO Evaluations

We fine-tuned OpenVLA via LoRA (r=32) with our OFT recipe on four LIBERO task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
In the initial version of our paper, we trained one checkpoint for each LIBERO task suite independently. In an updated version of the paper, we conducted an additional experiment in which we trained a single policy on all four task suites combined (results for this are available in the Additional Experiments section in the Appendix). Overall, the results for the task-specific policies and the combined policy are comparable: 97.1% vs. 96.8% average success rate across the four suites, respectively.

Below are the four independently trained OpenVLA-OFT checkpoints for LIBERO:
* [moojink/openvla-7b-oft-finetuned-libero-spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)
* [moojink/openvla-7b-oft-finetuned-libero-object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)
* [moojink/openvla-7b-oft-finetuned-libero-goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal)
* [moojink/openvla-7b-oft-finetuned-libero-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10)

Below is the OpenVLA-OFT checkpoint trained on all four task suites combined:
* [moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10)

To start evaluations with one of the independently trained checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above. You can set the `TRANSFORMERS_CACHE` and `HF_HOME` environment variable to change where the checkpoint files get cached.

我们使用LoRA（r=32）和我们的OFT配方对OpenVLA在四个LIBERO任务套件上进行了微调：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal和LIBERO-10（也称为LIBERO-Long）。在我们论文的最初版本中，我们为每个LIBERO任务套件独立训练了一个检查点。在论文的更新版本中，我们进行了一个额外的实验，即在所有四个任务套件的组合上训练了一个单一策略（该实验的结果可在附录的附加实验部分找到）。总体而言，任务特定策略和组合策略的结果是相当的：分别为97.1%和96.8%的平均成功率。

以下是针对LIBERO独立训练的四个OpenVLA-OFT检查点：

- moojink/openvla-7b-oft-finetuned-libero-spatial  
- moojink/openvla-7b-oft-finetuned-libero-object  
- moojink/openvla-7b-oft-finetuned-libero-goal  
- moojink/openvla-7b-oft-finetuned-libero-10  

以下是针对所有四个任务套件组合训练的OpenVLA-OFT检查点：

- moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10  

要使用其中一个独立训练的检查点开始评估，请运行以下命令之一。每个命令将自动下载上述列出的相应检查点。您可以通过设置`TRANSFORMERS_CACHE`和`HF_HOME`环境变量来更改检查点文件的缓存位置。
```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10
```

To evaluate the policy trained on all four task suites together, simply swap out the `--pretrained_checkpoint` in the commands above with `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`. There are
  other arguments in the script; we set them to the default values that work with the OpenVLA-OFT checkpoints above.
* **NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* The results reported in our paper were obtained using **Python 3.10.14, PyTorch 2.2.0, and our
  [custom transformers v4.40.1 fork](https://github.com/moojink/transformers-openvla-oft.git)**
  on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions if possible.
  Note that results may vary slightly if you use a different GPU than the A100. If the discrepancy is large,
  please post a GitHub issue, and we will look into it.

要评估在所有四个任务套件上一起训练的策略，只需将上述命令中的`--pretrained_checkpoint`替换为`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`即可。

注意事项：

- 评估脚本默认会运行500次试验（10个任务 × 每个任务50个剧集）。您可以通过设置`--num_trials_per_task`来修改每个任务的试验次数。您还可以通过`--seed`更改随机种子。脚本中还有其他参数，我们已将其设置为与上述OpenVLA-OFT检查点兼容的默认值。
- **注意**：设置`--center_crop True`非常重要，因为我们在训练时对OpenVLA进行了随机裁剪增强（我们在每个训练样本中随机裁剪了90%的区域，因此在测试时我们只需取中心90%的裁剪区域）。
- 评估脚本会在本地记录结果。您还可以通过设置`--use_wandb True`并将结果记录到Weights & Biases中，同时指定`--wandb_project <PROJECT>`和`--wandb_entity <ENTITY>`。
- 我们在论文中报告的结果是使用Python 3.10.14、PyTorch 2.2.0以及我们在NVIDIA A100 GPU上的自定义`transformers` v4.40.1版本获得的，并且是基于三个随机种子的平均值。如果可能，请尽量使用这些软件包版本。请注意，如果您使用的GPU不是A100，结果可能会略有不同。如果差异较大，请在GitHub上发布问题，我们将进行调查。
## Fine-Tuning on LIBERO Datasets

First, download the LIBERO datasets as mentioned above in the Setup section above: `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out).

Then, launch the fine-tuning script with the OFT configuration below, replacing `X` in the first line with the number of GPUs. The command below launches fine-tuning on LIBERO-Spatial with the hyperparameters that we used in our paper. Here, batch size 8 per GPU will require ~62 GB VRAM, and batch size 1 per GPU will require ~25 GB VRAM.

首先，按照上述“设置”部分提到的，下载LIBERO数据集：`libero_spatial_no_noops`、`libero_object_no_noops`、`libero_goal_no_noops`、`libero_10_no_noops`。（“_no_noops”表示没有无操作动作，即过滤掉了动作接近于零的训练样本）。

然后，使用以下OFT配置启动微调脚本，将第一行中的X替换为GPU的数量。以下命令将使用我们在论文中使用的超参数启动对LIBERO-Spatial的微调。在这里，每个GPU的批量大小为8将需要约62 GB的显存，而每个GPU的批量大小为1将需要约25 GB的显存。
```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

The above training command should reproduce our OpenVLA-OFT results if `X = 8` and the 150K step checkpoint is evaluated.

You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. You can also modify other args — e.g., if you want to train with just one input image from the third-person camera and disable proprio state input, you can set `--num_images_in_input 1` and `--use_proprio False`.

In general, we recommend fine-tuning until training L1 loss goes below 0.01 and starts to plateau (with the above configuration, it should reach ~0.006 L1 loss on LIBERO-Spatial after 150K gradient steps with 10x LR decay after 100K steps). However, for LIBERO-Goal only, we found that the 50K checkpoint (which was at ~0.02 L1 loss) performed best for unknown reasons. For all other task suites though, we found that the 150K checkpoint performed best.

Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially. You may be able to avoid the performance drop if you merge the LoRA weights into the base model on the downstream device used for testing (e.g., if you train on H100 and then merge on A100 before testing on A100). You can see our script [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) for merging the LoRA adapter into the base model offline. It's okay if you already merged LoRA weights into the base OpenVLA model during fine-tuning; you can always redownload the base model and merge again as long as you still have the LoRA adapter (`merge_lora_weights_and_save.py` will handle this for you).

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

如果X = 8，并且在150K步的检查点上进行评估，上述训练命令应该能够复现我们的OpenVLA-OFT结果。

你可以将`libero_spatial_no_noops`替换为`libero_object_no_noops`、`libero_goal_no_noops`或`libero_10_no_noops`。你也可以修改其他参数——例如，如果你想只使用第三人称摄像头的一个输入图像，并禁用本体状态输入，你可以设置`--num_images_in_input 1`和`--use_proprio False`。

一般来说，我们建议微调直到训练的L1损失降到0.01以下并开始趋于平稳（使用上述配置，它应该在150K梯度步后，在LIBERO-Spatial上达到约0.006的L1损失，并在100K步后将学习率降低10倍）。然而，对于LIBERO-Goal，我们发现50K检查点（其L1损失约为0.02）出于某种未知原因表现最好。对于所有其他任务套件，我们发现150K检查点表现最好。

请务必使用与训练时相同的设备/GPU来测试你的策略！否则，性能可能会大幅下降。如果你在下游设备上将LoRA权重合并到基础模型中（例如，如果你在H100上训练，然后在A100上合并，再在A100上进行测试），你或许可以避免性能下降。你可以查看我们的脚本`vla-scripts/merge_lora_weights_and_save.py`，用于离线将LoRA适配器合并到基础模型中。如果你已经在微调期间将LoRA权重合并到了基础OpenVLA模型中，没关系；只要你仍然有LoRA适配器，你可以随时重新下载基础模型并再次合并（`merge_lora_weights_and_save.py`会为你处理这个问题）。

如果你遇到任何问题，请新建一个GitHub问题。如果你在2个工作日内没有收到回复，请通过电子邮件联系Moo Jin Kim（moojink@cs.stanford.edu），以引起他的注意。