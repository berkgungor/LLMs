import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import Dataset, load_dataset
import csv
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk
import time

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

model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1
tokenizer.pad_token_id = 2
print(f"Successfully loaded the model {model_name} into memory")

def record_data(user_data, assistant_data):
    csv_file = 'ChatBot_Record_racoon.csv'
    column_names = ['User', 'Assistant']
    # Check if the file exists, and if not, write the column names
    try:
        with open(csv_file, 'r', newline='') as file:
            # File exists, no need to write column names
            pass
    except FileNotFoundError:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)

    # Open the CSV file in append mode
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the data to the CSV file
        writer.writerow([user_data, assistant_data])


from tkinter import scrolledtext, messagebox

def chatbot():
    def send_message():
        user_input = entry.get().strip()
        
        if user_input.lower() == 'exit':
            messagebox.showinfo("Chatbot", "Goodbye!")
            window.destroy()
            return
        
        # if user_input is not empty string
        if user_input:
            start_time = time.time()
            text = f"""<|system|>You are an AI assistant. Your job is to help your user specifically on golf exercises and training plans to improve their  skills.
                    <user>:{user_input}
                    <assistant>:"""

            model_input = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            
            eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
            generated_ids = model.generate(**model_input, max_new_tokens=100, do_sample=True, temperature=0.7,eos_token_id=eos_token_id)
            #decoded = tokenizer.batch_decode(generated_ids)
            #sentences = re.split(r'[.!?]|\\n|\n', decoded[0][len(text):])

            response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # get label from response text


            response_text = response_text[len(text):]
            print("res ",response_text)
            extracted_response = response_text.split("<user>")[0].strip()
            if "<<" in response_text:
                label = response_text.split("<<")[1].split(">>")[0].strip()
            else:
                label = "other"

            print("Exercise :", label)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken by the LLM: {time_taken} seconds")
            sentences = re.findall(r'[^.!?]*[.!?]', extracted_response)

            # Joining the sentences to form the trimmed response
            trimmed_response = ' '.join(sentences).strip()
            print(trimmed_response)
            
            assistant_data = trimmed_response
        else:
            assistant_data = ""

        chat_history.insert(tk.END, 'User :  ' + user_input + '\n')
        chat_history.insert(tk.END, ' Coach :  ' + assistant_data + '\n\n')
        chat_history.see(tk.END)

        chat_history.tag_configure("user", font=("Roboto", 14, "bold"))
        
        chat_history.tag_configure("assistant", font=("Roboto", 14, "bold"))
        record_data(user_input, assistant_data)
        entry.delete(0, tk.END)

    def clear_chat():
        chat_history.delete('1.0', tk.END)

    # Create the main window
    window = tk.Tk()
    window.title(" Coach GUI")
    window.configure(bg="#f0f0f0")

    original_image = Image.open('/home//Repository/AI_Coach/racoon.png') 
    resized_image = original_image.resize((470, 550), Image.Resampling.LANCZOS) 
    image = ImageTk.PhotoImage(resized_image)


    header_label = tk.Label(window, text=" Golf Coach", font=("Lato", 20, "bold"), bg="#f0f0f0")
    header_label.pack(side=tk.TOP, padx=10, pady=(20, 10))
    separator = tk.Frame(window, height=2, width=600, bg="#a6a6a6")
    separator.pack(pady=(20, 10))

    chat_history = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20,
                                            font=("Arial", 16), bg="#ffffff")
    chat_history.pack(fill=tk.BOTH, expand=True, padx=20,side=tk.LEFT)

    # Create a label to display the image on the right top
    image_label = tk.Label(window, image=image, bg="#f0f0f0")
    image_label.pack(side=tk.TOP, padx=5, pady=(5, 0))

    # Create a "Clean Chat" button to clear the chat history
    clean_chat_button = tk.Button(window, text="Clean Chat", command=clear_chat, font=("Roboto", 18), bg="green",
                              fg="#ffffff", activebackground="#0056b3", activeforeground="#ffffff")
    clean_chat_button.pack(side=tk.TOP, padx=10, pady=(10, 20))

    # Create a send button on the right bottom
    send_button = tk.Button(window, text="Send", command=send_message, font=("Roboto", 18), bg="#007bff",
                            fg="#ffffff", activebackground="#0056b3", activeforeground="#ffffff", height=2, width=10)
    send_button.pack(side=tk.BOTTOM, padx=10, pady=(10, 30))

    # Create an entry widget for user input on the right middle
    entry = tk.Entry(window, width=60, font=("Arial", 16))
    entry.pack(side=tk.BOTTOM, padx=10, pady=(10, 20))

    
    window.mainloop()

chatbot()


if __name__ == "__main__":
    chatbot()