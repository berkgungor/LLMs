
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

dataset_dir = "C:\\Users\\berkg\\source\\AI_Coach\\custom_orca_new.jsonl"
train_dataset = load_dataset('json', data_files=dataset_dir, split='train')
#eval_dataset = load_dataset('json', data_files='notes_validation.jsonl', split='train')

def formatting_func(example):
    text = f"{example['text']}"
    return text

base_model_id = "C:\\Users\\berkg\\source\\AI_Coach\\FineTune\\mistral_7b_instruct_sharded"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left", 
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)


import matplotlib.pyplot as plt

def plot_data_lengths(tokenize_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

plot_data_lengths(tokenized_train_dataset)