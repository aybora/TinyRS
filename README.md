# TinyRS-R1: Compact Vision Language Model for Remote Sensing

## IEEE Geoscience and Remote Sensing Letters

This paper presents TinyRS, the first 2B-parameter vision language models (VLMs) optimized for RS, and TinyRS-R1, its reasoning-augmented variant. Based on Qwen2-VL-2B, TinyRS is trained via a four-stage pipeline: pretraining on million-scale satellite images, instruction tuning, fine-tuning with chain-of-thought (CoT) annotations from a new reasoning dataset, and group relative policy optimization (GRPO)-based alignment. TinyRS-R1 matches or surpasses recent 7B RS models in classification, visual question answering (VQA), grounding, and open-ended QA—while using one third of the memory and latency. CoT reasoning improves grounding and scene understanding, while TinyRS excels at concise, low-latency VQA. TinyRS-R1 is the first domain-specialized small VLM with GRPO-aligned CoT reasoning for general-purpose RS.

### [Paper (arXiv)](https://arxiv.org/abs/2505.12099) | [Paper (IEEExplore)](https://ieeexplore.ieee.org/abstract/document/11207646)

### Dataset

<table>
  <thead>
    <tr style="text-align: right;">
      <th>dataset</th>
      <th>purpose</th>
      <th>link</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VHM_VersaD</td>
      <td>Pre-training</td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_VersaD">aybora/VHM_VersaD</a></td>
      <td>Duplicated from <a href="https://huggingface.co/datasets/FitzPC/VHM_VersaD">FitzPC/VHM_VersaD</a> and added list_pretrain_corrected.json including corrected folder structure used in this codebase.</td>
    </tr>
    <tr>
      <td>VHM_dataset_sft</td>
      <td>SFT</td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_sft">aybora/VHM_dataset_sft</a></td>
      <td>Duplicated from <a href="https://huggingface.co/datasets/FitzPC/VHM_dataset_sft">FitzPC/VHM_dataset_sft</a> and added list_sft_oversampled.json and list_sft_reasoning_oversampled.json files for oversampled and CoT captioning for generation of TinyRS abd TinyRS-CoT.</td>
    </tr>
    <tr>
      <td>VHM_dataset_grpo</td>
      <td>GRPO Training</td>
      <td><a href="https://huggingface.co/datasets/aybora/VHM_dataset_grpo">aybora/VHM_dataset_grpo</a></td>
      <td>RL dataset for TinyRS-R1</td>
    </tr>
    <tr>
      <td>scorers_datasets_eval</td>
      <td>Evaluation</td>
      <td><a href="https://huggingface.co/datasets/aybora/scorers_datasets_eval">aybora/scorers_datasets_eval</a></td>
      <td>Duplicated from <a href="https://huggingface.co/datasets/LHRS/RSRM">LHRS/RSRM</a> and prompts made applicable for CoT and codebase.</td>
    </tr>
  </tbody>
</table>

### Models

<table>
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>type</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TinyRS-PRETRAIN</td>
      <td>pretraining</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-TinyRS-PRETRAIN">aybora/Qwen2-VL-TinyRS-PRETRAIN</a></td>
    </tr>
    <tr>
      <td>TinyRS</td>
      <td>instant</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-TinyRS">aybora/Qwen2-VL-TinyRS</a></td>
    </tr>
    <tr>
      <td>TinyRS-CoT</td>
      <td>reasoning (sft only)</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-TinyRS-CoT">aybora/Qwen2-VL-TinyRS-CoT</a></td>
    </tr>
    <tr>
      <td>TinyRS-R1</td>
      <td>reasoning</td>
      <td><a href="https://huggingface.co/aybora/Qwen2-VL-TinyRS-R1">aybora/Qwen2-VL-TinyRS-R1</a></td>
    </tr>
  </tbody>
</table>

### Installation

For best reproducibility, we suggest you to generate three different environments, one each for finetuning, GRPO training and evaluation.

For SFT:

```shell
git clone https://github.com/aybora/TinyRS
conda env create -f environment.yaml
conda activate sft
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

For GRPO:

```shell
git clone https://github.com/aybora/TinyRS
conda create -n grpo python=3.10 -y
conda activate grpo
cd ~/TinyRS/grpo
pip3 install -e ".[dev]"
pip3 install wandb==0.18.3
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
```

For Evaluation:

```shell
conda create -n eval python==3.10 -y
conda activate eval
cd ~/TinyRS/eval 
bash basic_env_setup.sh
```

### Training (SFT)

For pre-training, SFT and CoT SFT, you may follow the sample script below, which works on one node with 4 x H100s or A100s. Sample script assumes you are doing a pretraining, change data_path and image_folder otherwise.

For SFT, change model name to aybora/Qwen2-VL-TinyRS-PRETRAIN, image_folder to ../VHM_dataset_sft, data_path to ../VHM_dataset_sft/list_sft_oversampled.json

For CoT SFT, change model name to aybora/Qwen2-VL-TinyRS, image_folder to ../VHM_dataset_sft, data_path to ../VHM_dataset_sft/list_sft_reasoning_oversampled.json

```shell

conda activate sft
cd ./TinyRS/sft/

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed --master_port 29400 src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path ../VHM_VersaD/list_pretrain_corrected.json \
    --image_folder ../VHM_VersaD \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen_vhm_pretrain_2b \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --dataloader_num_workers 4
```
### Training (GRPO)

Below script works on at least one node with 4 x H100s or A100s (65-80 GB). Please note that, this script requires OPENAI_API_KEY for exact reproduction of the model, which may cost ~10-15$ per full training. You may skip open ended questions by adding a flag or removing questions with answer_type=0 from the dataset, if you don't have an API key.

```shell
export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export OPENAI_API_KEY=#ENTER YOUR API KEY

torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id $SLURM_JOB_ID \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path aybora/Qwen2-VL-TinyRS-CoT \
    --dataset_name aybora/VHM_dataset_grpo \
    --max_prompt_length 8192 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.001 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 6 \
    --num_train_epochs 64 \
    --num_generations 4 \
    --save_steps 100 \
    --run_name $WANDB_RUN_NAME
```
You may need to adjust some of the parameters (MASTER_ADDR, GPUS_PER_NODE etc.) depending on your multi-gpu, multi-node setting. 

### Evaluation

First download datasets eval folder from our [Huggingface Repo](https://huggingface.co/datasets/aybora/scorers_datasets_eval), which is forked from [ScoreRS HF Repo](https://huggingface.co/datasets/LHRS/RSRM/tree/main).

To evaluate our, or your reproduced model, you may use the script below:

```shell

SCRIPT_PATH=./TinyRS/eval/python_script/evaluation/rs_evaluation.py
DATA_ROOT="Your path to datasets eval folder"
OUTPUT_DIR="Your path to eval log file"
model_type=lmdeploy
MODEL_PATH=aybora/Qwen2-VL-TinyRS-R1
REASONING_CONFIG=./TinyRS/eval/config/qwen2_thinking_template.json

PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 $SCRIPT_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --model_type $model_type \
    --model_path $MODEL_PATH \
    --force_inference true \
    --task all \
    --reasoning_config $REASONING_CONFIG

```

### Acknowledgements

Our work is derived from [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model, [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) for the forked main sft code, [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) for the forked main grpo code, [ScoreRS](https://github.com/NJU-LHRS/ScoreRS) for the forked main evaluation code and [VHM](https://github.com/opendatalab/VHM) for the dataset and the base captions. We appreciate all of these great works.

### Citation

If you find this code useful for your research, consider citing our works:

```bibtex
@article{koksal2025tinyrs,
  title={Tinyrs-r1: Compact vision language model for remote sensing},
  author={K{\"o}ksal, Aybora and Alatan, A Ayd{\i}n},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2025},
  publisher={IEEE}
}
```

```bibtex
@ARTICLE{koksal2025samchat,
  author={Köksal, Aybora and Alatan, A. Aydın},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={SAMChat: Introducing Chain-of-Thought Reasoning and GRPO to a Multimodal Small Language Model for Small-Scale Remote Sensing}, 
  year={2026},
  volume={19},
  number={},
  pages={795-804},
  keywords={Cognition;Adaptation models;Computational modeling;Remote sensing;Visualization;Missiles;Satellite images;Mathematical models;Large language models;Analytical models;Aerial image analysis;chain-of-thought (CoT) reasoning;domain adaptation;group relative policy optimization (GRPO);multimodal large language models (MLLMs);remote sensing (RS)},
  doi={10.1109/JSTARS.2025.3637115}}
}
```

If you are interested in this work, you may find the following work also useful:

```bibtex
@inproceedings{koksal2025few,
  title={Few-Shot Vision-Language Reasoning for Satellite Imagery via Verifiable Rewards},
  author={K{\"o}ksal, Aybora and Alatan, A Ayd{\i}n},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6901--6910},
  year={2025}
}
```
