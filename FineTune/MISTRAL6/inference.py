from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import time
import torch
import re 

base_model = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model)
adapters_name = "/home//Repository/AI_Coach/MISTRAL6/results_mistral_v7/checkpoint-200"
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model_reload, adapters_name)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def inf(user_prompt):
    start_time = time.time()
    B_INST, E_INST = "[INST]", "[/INST]"
    prompt = f"{B_INST}{user_prompt.strip()}\n{E_INST}"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, max_length=150)
    #_ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)
    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7,eos_token_id=2)
    #output = streamer.process(generated_ids)[0]
    response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response_text = response_text[len(prompt):]
    extracted_response = response_text.split("[INST]")[0].strip()
    if "<<" in response_text:
                label = response_text.split("<<")[1].split(">>")[0].strip()
    else:
        label = "other"
    
    print(extracted_response.split("<<")[0].strip())
    print("Exercise :", label)

    print(f"Time taken: {time.time() - start_time:.2f}s")
    


while True:
    user_prompt = input("User: ")
    inf(user_prompt)
    print("\n")

