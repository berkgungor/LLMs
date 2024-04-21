from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Assuming TEMPLATE1, TEMPLATE2, TEMPLATE3, test_input and test_output are defined elsewhere

test_prompts = [
    "The conversation between Human and AI Golf assistant named Coach [INST] Are you good at golf? [/INST]",
]

test_output = "Yes, I am pretty good at indoor golf. That's why I am here to help you."

# Load your baseline model and tokenizer
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
baseline = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Loop through each prompt to generate output and compare with the expected output
for prompt in test_prompts:

    encoded_prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = baseline.generate(
        **encoded_prompt,
        max_new_tokens=200, # set accordingly to your test_output
        do_sample=False
    )

    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Output results for comparison
    print(f"Generated Output: {decoded_output}\n")
    print(f"Expected Output: {test_output}\n")
    print("-" * 75)