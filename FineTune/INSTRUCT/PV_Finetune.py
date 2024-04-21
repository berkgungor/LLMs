from datasets import load_dataset
import torch
import accelerate
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
# download dataset
torch.cuda.empty_cache()

dataset_dir = "Data//Data_Finetune_Mistral.jsonl"
dataset = load_dataset('json', data_files=dataset_dir, split='train')

def format_instruction(sample):
    return f"""You are a personal assistant  and you also enlight users about  products. Help users to learn  fundamentals and techniques. Recommend ideal  exercises to help users to improve their  skills.
        ### Instruction:{sample["instruction"]} ### Response: {sample["response"]}"""