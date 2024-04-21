from peft import AutoPeftModelForCausalLM,PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, TextStreamer
from transformers import AutoTokenizer
import torch
import time
from datasets import load_dataset
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_dir = "//home////Repository//AI_Coach//INSTRUCT//Data_Finetune_Mistral.jsonl"
dataset = load_dataset('json', data_files=dataset_dir, split='train')

base_model = "mistralai/Mistral-7B-v0.1"
adapters_name = "/home//Repository/AI_Coach/INSTRUCT/mistral-7b-golf-assistant5"



bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)

model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_instruction(sample):
    return f"""You are a personal assistant . Help users to learn  fundamentals and techniques. Recommend ideal  exercises to help users to improve their  skills.
        ### Instruction:{sample} ### Response:"""


def call_inference(sample):

    #sample = dataset[randrange(len(dataset))]
    prompt = format_instruction(sample)
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
    with torch.inference_mode():
        outputs = model.generate(
            **input_ids, 
            max_new_tokens=200, 
            do_sample=True, 
            top_p=0.9,
            temperature=0.7
        )

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("full output :", outputs)
    output = outputs[0][len(prompt):]
    #print("output :", output)

    if output:
        if "### Instruction:" in output:
            output = output.split("### Instruction:")[0].strip()

        instruction = sample
        output = output
        groud_truth = ""#sample['response']

        print(f"Instruction: \n{instruction}\n\n")
        #print(f"Ground truth: \n{groud_truth}\n")
        print(f"Generated output: \n{output}\n\n\n")   
    else:
        print("I am sorry, I did not understand that. Could you please rephrase your question?")
    #return instruction, output, groud_truth

while True:
    prompt = input("User: ")
    call_inference(prompt)