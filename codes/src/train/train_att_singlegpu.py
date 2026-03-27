import os
import torch
import json
import random
import numpy as np
import time
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List

from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from datasets.alltok_dataset import get_attribute_dataset


MODEL_PATH = "your_path/Qwen/Qwen2.5-VL-7B-Instruct"
JSON_PATH = "your_data/demo_attribute.json"
IMAGE_DIR = "your_data"
TOKEN_FILE = "your_data/attribute_list.txt"
BASE_OUTPUT_DIR = "outputs/attribute_training"
BATCH_SIZE = 5
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
gpu_id = 0

def train_worker(seed):
    # 防止多进程同时下载/读取产生冲突，做错峰
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 模型加载
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    torch.cuda.set_device(gpu_id) # 强制当前进程使用该卡
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)

    # 3. LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=r".*visual.*",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()
    
    # 4. 数据加载
    # 1. Processor 配置
    train_loader , dataset, processor = get_attribute_dataset(MODEL_PATH, JSON_PATH, IMAGE_DIR, TOKEN_FILE, batch_size=BATCH_SIZE)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * max_train_steps),
        num_training_steps=max_train_steps
    )
    
    global_step = 0
    optimizer.zero_grad()

    # 5. 训练循环
    for epoch in range(NUM_EPOCHS):
        if gpu_id == 0:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[GPU{gpu_id}] Epoch {epoch+1}", position=gpu_id)
        
        for step, batch in enumerate(pbar):
            inputs = {}
            for k, v in batch.items():
                if k == "pixel_values":
                    inputs[k] = v.to(device, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
                
            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / num_update_steps_per_epoch
        print(f"[GPU{gpu_id}] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Save Checkpoints
        epoch_path = os.path.join(OUTPUT_DIR, f"lora_epoch-{epoch+1}")
        model.save_pretrained(epoch_path)
        processor.save_pretrained(epoch_path)

    final_path = os.path.join(OUTPUT_DIR, "lora_final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)


if __name__ == "__main__":
    SEED = 42
    train_worker(SEED)