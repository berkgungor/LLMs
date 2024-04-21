import os
from random import shuffle
import transformers, torch
import wandb
from datasets import load_dataset
import transformers
import pandas as pd

PROJECT_DIR_PATH = ''
OUTPUT_PATH = os.path.join(PROJECT_DIR_PATH, 'models')

BASELINE_MODEL_NAME = 'HuggingFaceH4/zephyr-7b-beta'
PROJECT_NAME = 'zephyr-7b-beta_assistant_v0.2'
#HUGGING_FACE_REPO_NAME = f'Pazuzzu/{PROJECT_NAME}'
#HUGGING_FACE_MERGED_REPO_NAME = f'{HUGGING_FACE_REPO_NAME}_merged'
#HUGGING_FACE_GPTQ_REPO_NAME = f'{HUGGING_FACE_REPO_NAME}_gptq'


wandb.login()
wandb.init(project=PROJECT_NAME)

# Load Dataset and convert to pandas.
dataset_dir = "C:\\Users\\berkg\\source\\AI_Coach\\Data\\converted_data.json"
ds = load_dataset('json', data_files=dataset_dir, split='train')
df = ds['train'].to_pandas()

tokenizer = transformers.AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME, padding_side='left',add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token


# Define cleaning and preprocessing functions.
def text_to_dialogue(text):
    return [sentence.replace('User:', '').replace('Chip:', '').strip() for sentence in text.split('Assistant:')]

def dialogue_to_chat(dialogue):
    out = [{'role': 'system', 'content': 'You are a friendly chatbot assistant.'}]
    for idx, message in enumerate(dialogue):
        role = 'user' if idx%2==0 else 'assistant'
        out.append({'role': role, 'content': message})
    return out

def chat_to_input(chat):
    return tokenizer.apply_chat_template(chat, tokenize=False)

def process_example(example):
    out = text_to_dialogue(example)             # Clean up data from redundant words
    out = dialogue_to_chat(out)                 # Convert to ChatML format
    out = chat_to_input(out)                    # Add a task description & Apply base model chat template
    return out

# Apply on dataset.
data = list(map(process_example, df['text']))

# Shuffle dataset.

shuffle(data)

# Tokenize data.
tokenized_data = list(map(tokenizer, data))

split_idx = int(.99 * len(data))
train_data, val_data = tokenized_data[:split_idx], tokenized_data[split_idx:]


# Get quantized model.
model = transformers.AutoModelForCausalLM.from_pretrained(BASELINE_MODEL_NAME,
                                                          load_in_8bit=True,     # call for the 8 bit bnb quantized version
                                                          device_map='auto')
# Get tokenizer.
tokenizer = transformers.AutoTokenizer.from_pretrained(HUGGING_FACE_REPO_NAME,
                                                       padding_side='left',      # Use left padding for open end generation
                                                       add_eos_token=True)       # Add pad token as many llm tokenizers don't have it setup by default
tokenizer.pad_token = tokenizer.eos_token

# Set PEFT adapter config (16:32).
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],                     # Apply to "q_proj", "v_proj" layers of attention as suggested by paper
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")

# Stabilize output layer and layernorms & prepare for 8bit training.
model = prepare_model_for_kbit_training(model, 8)

# Set PEFT adapter on model.
model = get_peft_model(model, config)

#Zephyr 7B Beta is already finetuned for the assistance task, so it will be hard to finetune it further, so we’ll be using an early stopping callback and enabling save_best flag in order to prevent overtraining:

# Set Hyperparameters.
MAXLEN=512
BATCH_SIZE=6
GRAD_ACC=4
WARMUP=100
STEPS=1000
OPTIMIZER='paged_adamw_8bit'                                # Use paged optimizer to save memory
LR=4e-5                                                     # Use value slightly smaller than pretraining lr value & close to LoRA standard

# Setup Callbacks.
early_stop = transformers.EarlyStoppingCallback(10, 1.15)

# Set training config.
training_config = transformers.TrainingArguments(per_device_train_batch_size=BATCH_SIZE,
                       gradient_accumulation_steps=GRAD_ACC,
                       warmup_steps=WARMUP,
                       max_steps=STEPS,
                       optim=OPTIMIZER,
                       learning_rate=LR,
                       fp16=True,                            # Consider using bf16 if compatible with your GPU
                       logging_steps=1,
                       output_dir=OUTPUT_PATH,
                       report_to='wandb',
                       load_best_model_at_end=True,
                       evaluation_strategy='steps',
                       metric_for_best_model='eval_loss',
                       greater_is_better=False,
                       eval_steps=10,
                       save_steps=10,
                       save_total_limit=2)

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Setup trainer.
trainer = transformers.Trainer(model=model,
                               train_dataset=train_data,
                               eval_dataset=val_data,
                               data_collator=data_collator,
                               args=training_config,
                               callbacks=[early_stop])

model.config.use_cache = False  # Silence the warnings.
trainer.train()

# Push to HF Hub.
#model.push_to_hub(HUGGING_FACE_REPO_NAME)
#tokenizer.push_to_hub(HUGGING_FACE_REPO_NAME)

from peft import PeftConfig
def merge():
    # Get peft config.
    config = PeftConfig.from_pretrained(HUGGING_FACE_REPO_NAME)

    # Get base model
    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                            torch_dtype=torch.float16,      # GPTQ quantization requires fp16
                                                            return_dict=True)

    # Load the Lora model.
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, HUGGING_FACE_REPO_NAME)
    merged_model = model.merge_and_unload()
    # Push to HF Hub.
    merged_model.push_to_hub(HUGGING_FACE_MERGED_REPO_NAME)
    tokenizer.push_to_hub(HUGGING_FACE_MERGED_REPO_NAME)
    
    
def quantize():
    torch.cuda.empty_cache()
    quantization_config = transformers.GPTQConfig(bits=4,
                                                group_size=128,
                                                desc_act=False,
                                                dataset=data[:660],    # Use data used for training for calibration
                                                tokenizer=tokenizer)
    # Quantize Model
    q_model = transformers.AutoModelForCausalLM.from_pretrained(HUGGING_FACE_MERGED_REPO_NAME, 
                                                                quantization_config=quantization_config, 
                                                                torch_dtype=torch.float16, 
                                                                device_map="auto")       # If not device_map not supported, used offload_folder
    
        # Push to HF hub.
    q_model.push_to_hub(HUGGING_FACE_GPTQ_REPO_NAME)
    tokenizer.push_to_hub(HUGGING_FACE_MERGED_REPO_NAME)
    
    


def inference():
    # Get model.
    model = transformers.AutoModelForCausalLM.from_pretrained(HUGGING_FACE_MERGED_REPO_NAME,
                                                            device_map="auto",
                                                            torch_dtype=torch.bfloat16,
                                                            use_flash_attention_2=True)   # Use flash attention for faster inference
                                                # Use low_cpu_mem_usage=True if you run into memory issues.

    # Get tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(HUGGING_FACE_MERGED_REPO_NAME, padding_side='left')
    # Try GPTQ model.
    messages = [
        {"role": "system", "content": "You are a friendly chatbot assistant."},
        {"role": "user", "content": "Hello, what are the keys things to keep in mind during a job interview ?"},]
    gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')

    gen_output = model.generate(input_ids=gen_input,
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50, 
                                top_p=0.95, 
                                repetition_penalty=1.1)
    out = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(out)