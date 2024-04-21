from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
chat = [
  {"system_prompt": "You are an AI Golf coach that helps people to train and find information about golf.", "question": "Who won the World Chess Championship in 2021?", "response": "Magnus Carlsen defeated Ian Nepomniachtchi 7.5 - 3.5 to become the 2021 World Chess Champion."},
]

tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))