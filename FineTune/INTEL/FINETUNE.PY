import os
from copy import deepcopy
from random import randrange
from functools import partial
import torch
import accelerate
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

import wandb
wandb.login()

model_name = "Intel/neural-chat-7b-v3-1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
new_model = "Intel/neural-golf-7b-v3-1_v1"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)


model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)


lora_alpha = 16
lora_dropout = 0.1
lora_r = 8


peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=modules,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, peft_config)


### LOAD DATASET

dataset_dir = "//home////Repository//AI_Coach//modified_dataset_new_format.json"
dataset = load_dataset('json', data_files=dataset_dir, split='train')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_prompt(sample):

    INTRO = "<|system|>You are a Golf Coach. Your job is to help your user to improve their  skills with providing exercises and training plans."
    END = "Instruction: Provide  exercises or training plans based on user's request or write a response appropriate to the conversation."

    conversations = ""
    for response in sample["conversations"]:
      from_, value = response["from"], response["value"]
      conversations += f"<{from_}>: " + value + "\n"

    sample["text"] = "\n\n".join([INTRO, conversations])
    return sample


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str, seed: int = 42):
    # Format each prompt.
    print("Preprocessing dataset...")
    dataset = dataset.map(format_prompt)

    def preprocess_batch(batch, tokenizer, max_length):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )


    # Apply preprocessing to each batch of the dataset & and remove "conversations" and "text" fields.
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["conversations", "text"],
    )
    # Shuffle dataset.
    dataset = dataset.shuffle(seed=seed)

    return dataset

max_length = 512
dataset = preprocess_dataset(tokenizer, max_length, dataset)


#=========================== WANDB ==========================================
run = wandb.init(
    project="finetuning_neural-chat",
    name="log_dataset",
)

dataset.save_to_disk("GolfInstruct_neural-chat.hf")
artifact = wandb.Artifact(name="GolfInstruct_neural-chat", type="dataset")
artifact.add_dir("./GolfInstruct_neural-chat.hf", name="train")
run.log_artifact(artifact)

run.finish()

run = wandb.init(
    project="finetuning_neural-chat",   # Project name.
    name="run0",                     # name of the run within this project.
)
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log model checkpoints.


#=========================== TRAINING ==========================================
training_args = TrainingArguments(
    output_dir="neural-chat_outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    max_steps=1500,
    lr_scheduler_type="linear",
    warmup_steps=5,
    fp16=True,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    optim="paged_adamw_8bit",
    report_to="wandb"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset,
)

results = trainer.train()  # Now we just run train()!
trainer.model.save_pretrained(new_model)
run.finish()


### Merge model

def merge_model():
    # Merge the model with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model= PeftModel.from_pretrained(base_model, new_model)
    merged_model= merged_model.merge_and_unload()
    torch.cuda.empty_cache()
    # Save the merged model
    merged_model.save_pretrained("zephyr_beta_merged_model_v4",safe_serialization=True)
    tokenizer.save_pretrained("zephyr_beta_merged_model_v4")
    print("========= merged Model saved =========")
    return merged_model, tokenizer

#merge_model()
