from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import time
import torch
import re 

base_model = "HuggingFaceH4/zephyr-7b-beta"
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)

adapters_name = "/home//Repository/AI_Coach/ZEPHYR/results_zephyr_v8/checkpoint-300"
model = PeftModel.from_pretrained(base_model_reload,  adapters_name )
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
pipe = pipeline("text-generation", model=model,tokenizer=tokenizer,max_length=200 ,torch_dtype=torch.bfloat16, device_map="auto")

def inf(user_prompt):
    prompt = user_prompt
    start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": "You are a personal assistant . help users to learn  fundamentals and improve their  skills with ideal exercises",
        },
        {"role": "user", "content": prompt},
    ]
    #print(messages)
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    assistant = outputs[0]["generated_text"].split("<|assistant|>")[1].strip()
    assistant = assistant.split("<")[0].strip()
    print("\n")
    print(assistant)
    print(f"Time taken: {time.time() - start_time:.2f}s")

while True:
    user_prompt = input("User: ")
    inf(user_prompt)
    print("\n")

