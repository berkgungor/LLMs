
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers import pipeline

model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
adapters_name = "/home//Repository/AI_Coach/ZEPHYR/ZEPHYR_outputs_beta_v8/checkpoint-900"



device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)

model1 = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model1,tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")

def response_zephyr(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are a personal assistant . Your job is to answer questions and only if users ask something about golf then help users to learn  fundamentals and improve their  skills with ideal exercises",
        },
        {"role": "user", "content": user_input},
    ]
    #print(messages)
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return (outputs[0]["generated_text"])

def generate_response(system_input, user_input):

    # Format the input using the provided template
    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

    # Generate a response
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    return response.split("### Assistant:\n")[-1]


# Example usage
while True:
    system_input = "You are a personal assistant and your specialisation is indoor  practice. Your job is to answer questions and only if users ask something about golf then help users to learn  fundamentals and improve their  skills with ideal exercises."
    user_input = input("User: ")
    #response = generate_response(system_input, user_input)
    response = response_zephyr(user_input)
    print(response.split("<|assistant|>")[-1])
