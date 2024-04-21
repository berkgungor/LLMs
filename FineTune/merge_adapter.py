import os
from copy import deepcopy
from random import randrange
from functools import partial
import torch
import accelerate
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

model_name = "mistralai/Mistral-7B-v0.1"
adapter = "C:\\Users\\berkg\Repository\\-ai-coach\\model_v1"

def merge_model():
    # Merge the model with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model= PeftModel.from_pretrained(base_model, adapter)
    merged_model= merged_model.merge_and_unload()
    torch.cuda.empty_cache()
    # Save the merged model
    merged_model.save_pretrained("mistral_golf_assistant_v1",safe_serialization=True)
    tokenizer.save_pretrained("mistral_golf_assistant_v1")
    print("========= merged Model saved =========")
    return merged_model, tokenizer

merge_model()