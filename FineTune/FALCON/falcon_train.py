import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
    TrainingArguments
)
from peft.tuners.lora import LoraLayer
from trl import SFTTrainer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_name = "vilsonrodrigues/falcon-7b-instruct-sharded"
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value"
    ],
)


dataset = load_dataset("berkouille/GolfData", split="train")

training_arguments = TrainingArguments(
    output_dir="./training_results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)

trainer.train()

trained_model_dir = "./training_model"
model.save_pretrained(trained_model_dir)


from peft import PeftConfig, PeftModel
# # Inference
def inference():
    config = PeftConfig.from_pretrained(trained_model_dir)

    trained_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_namme_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    trained_model = PeftModel.from_pretrained(trained_model, trained_model_dir)
    trained_model_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path , trust_remote_code=True)
    trained_model_tokenizer.pad_token = trained_model_tokenizer.eos_token