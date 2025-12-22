# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import ast
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from PIL import Image
import base64
import io
from openai import OpenAI

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

#from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

def my_load_dataset(dataset_path, split=None):
    if os.path.isdir(dataset_path):
        print(f"✅ Loading dataset from local disk: {dataset_path}")
        return load_from_disk(dataset_path)
    else:
        print(f"🌐 Local path not found. Loading from Hugging Face Hub: {dataset_path}")
        return load_dataset(dataset_path, split=split)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def parse_bbox(raw: str):
    # Pure-regex parser: robust to leading zeros, stray brackets, etc.
    nums = list(map(int, re.findall(r'-?\d+', raw)))
    if len(nums) < 4:
        raise ValueError(f"Need 4 coords in: {raw}")
    return nums[:4]          # [x1, y1, x2, y2]

def iou_from_strings(student_answer: str, ground_truth_str: str) -> float:
    s_bbox = parse_bbox(student_answer)      # ← new names
    g_bbox = parse_bbox(ground_truth_str)

    # Intersection
    x1, y1 = max(s_bbox[0], g_bbox[0]), max(s_bbox[1], g_bbox[1])
    x2, y2 = min(s_bbox[2], g_bbox[2]), min(s_bbox[3], g_bbox[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Area & IoU
    area_s = (s_bbox[2] - s_bbox[0]) * (s_bbox[3] - s_bbox[1])
    area_g = (g_bbox[2] - g_bbox[0]) * (g_bbox[3] - g_bbox[1])
    union = area_s + area_g - inter
    return inter / union if union else 0.0

def accuracy_reward(completions, solution, **kwargs):
    """Custom reward function based on keyword presence and a solution flag."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    OPENAI_PROMPT = """ You are a teacher who is checking the answer of a student model. The student is trying to analyze the given image answer the given question. 
    Since these will get open ended answers, there are not correct responses. However, we can still grade its quality. 
    I would like you to analyze this image and analyze the reponse of my model and grade it between 0-10, based on its explanatory abilities, quality of responses, choice of words etc. 
    Your only response should be a number between 0 to 10. 
    Question will be given in the <question> </question> tags. Answer of the student model will be given in the <answer> </answer> tags.
"""

    for content, sol in zip(contents, solution):
        reward = 0.0
        answer_type = sol[0]
        answer = sol[1]
        question = sol[2]
        image = sol[3]

        # Extract the text within <answer> tags; if not found, use the full content.
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        student_answer = answer_match.group(1).strip() if answer_match else content.strip()

        if answer_type == "0": #open ended
            # Construct your detailed prompt
            gpt_question = f""" <question> {question} </question> <answer> {student_answer} </answer> """
            # Prepare messages for the ChatCompletion API
            messages = [
                {"role": "system", "content": OPENAI_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": gpt_question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                        },
                    ],
                }
            ]
            # Call the OpenAI ChatCompletion API
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",  # or use "gpt-3.5-turbo" if GPT-4 is not available
                    messages=messages,
                    temperature=0.2,
                    max_tokens=32768  # Adjust token limit based on your needs and model context size
                )
            except Exception as e:
                print(f"API error: {e}")
                reward = 0.0
            generated_text = response.choices[0].message.content
            # Extract the score from the generated text.
            score_match = re.search(r"\d+(\.\d+)?", generated_text)
            if score_match:
                score = float(score_match.group(0))
                # Normalize score to [0.0, 1.0] range
                reward = max(0.0, min(score / 10.0, 1.0))
            else:
                reward = 0.0

        elif answer_type == "1": #ground truth
            ground_truth = answer.strip()
            # Compare the student's answer with the ground truth.
            if student_answer.lower().replace('.', '') == ground_truth.lower().replace('.', ''):
                reward = 1.0
        elif answer_type == "2": #bounding box
            try: 
                nums = [int(n) for n in re.findall(r'\d+', student_answer)]
                student_answer = [nums[i:i+4] for i in range(0, len(nums), 4)][0]
                
                nums_gt = [int(n) for n in re.findall(r'\d+', answer)]
                ground_truth = [nums[i:i+4] for i in range(0, len(nums_gt), 4)][0]
                # Calculate the intersection over union (IoU) between the two bounding boxes.
                x1 = max(student_answer[0], ground_truth[0])
                y1 = max(student_answer[1], ground_truth[1])
                x2 = min(student_answer[2], ground_truth[2])
                y2 = min(student_answer[3], ground_truth[3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area_student = (student_answer[2] - student_answer[0]) * (student_answer[3] - student_answer[1])
                area_ground_truth = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
                union = area_student + area_ground_truth - intersection
                iou = intersection / union if union > 0 else 0
                reward = iou
            except Exception as e:
                print(f"IoU error: {e}")
                reward = 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution flag: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., "
    "<reasoning> reasoning process here </reasoning><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = my_load_dataset(dataset_path=script_args.dataset_name)

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
