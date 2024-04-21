from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")

# Load your custom dataset
train_dataset = load_dataset('your_custom_dataset', split='train')
eval_dataset = load_dataset('your_custom_dataset', split='validation')

# Tokenize the datasets
def tokenize_function(examples):
   return tokenizer(examples["input_text"], truncation=True, padding="max_length")

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
   output_dir="./results",
   num_train_epochs=1,
   per_device_train_batch_size=4,
   per_device_eval_batch_size=4,
   gradient_accumulation_steps=1,
   gradient_checkpointing=True,
   max_grad_norm=0.3,
   learning_rate=2e-4,
   weight_decay=0.001,
   optim="paged_adamw_32bit",
   lr_scheduler_type="constant",
   max_steps=-1,
   warmup_ratio=0.03,
   group_by_length=True,
   save_steps=25,
   logging_steps=25,
)

# Initialize the Trainer
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()