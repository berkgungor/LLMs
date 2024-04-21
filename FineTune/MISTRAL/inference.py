import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import Dataset, load_dataset

adapters_name = "mistralai-Golf-Instruct_axelle"
model_name = "../mistral_7b_instruct_sharded" #"mistralai/Mistral-7B-Instruct-v0.1"
train_dataset = load_dataset('json', data_files='../../Data/mistral_instruction_format_train.jsonl' , split='train')

device = "cuda:0"  # Set the desired GPU device

# Define the custom device map
device_map = {
    'cuda:0': 'cuda',  # Map 'cuda:0' to 'cuda' to use the GPU
}
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit=False,
    device_map=device_map 
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit_fp32_cpu_offload=True, 
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map=device_map
)

model = PeftModel.from_pretrained(model, adapters_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1
tokenizer.pad_token_id = 2

print(f"Successfully loaded the model {model_name} into memory")

def inference1():
    sample = train_dataset [-263]
    text = f"""<s>[INST]
    {sample['instruction']}
    [/INST]""".strip()

    model_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    generated_ids = model.generate(**model_input, max_new_tokens=50, do_sample=True,temperature=0.9)
    decoded = tokenizer.batch_decode(generated_ids)
    print("------------------- User -------------------------")
    print(sample['instruction'])
    print("----------------- Assistant -------------------------")
    #print(decoded[0][len(text):].split('.')[0], '.')
    # print first two senteces of the generated text
    print(decoded[0][len(text):].split('.')[0], '.', decoded[0][len(text):].split('.')[1], '.')
    print("---------------------------------------------")
    print(f"Ground truth:\n{sample['output']}")
    #print(decoded)

inference1()