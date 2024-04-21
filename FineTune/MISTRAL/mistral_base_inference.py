import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

device = "cuda" # the device to load the model onto


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "bn22/Mistral-7B-Instruct-v0.1-sharded",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)

text = "<s>[INST] can you recommend me a  exercise to improve my direction accuracy? [/INST]"

encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)

model_inputs = encodeds

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])