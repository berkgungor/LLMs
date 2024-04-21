from llama_cpp import Llama
from tkinter import scrolledtext, messagebox, PhotoImage, Tk, Label, Frame, Button, Entry, END, WORD
from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
import csv
import time 


class Chatbot:
    def __init__(self, model_path):
        self.llama = Llama(model_path=model_path)

    def get_response(self, input_text):
        old_text = f"""<|system|>You are an AI assistant. Your job is to help your user specifically on golf exercises and training plans to improve their  skills.handle the conversation history with care.
                <|user|>:{input_text}
                <|assistant|>:"""
        output = self.llama(old_text, max_tokens=120, temperature=0.7, top_p=0.5, echo=False, stop="<user>:")
        return output["choices"][0]["text"]
    
    def record_data(self, user_data, assistant_data):
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



class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[2].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        
        

class ChatbotGUI:
    def __init__(self, chatbot, tts_engine):
        self.chatbot = chatbot
        self.tts_engine = tts_engine
        self.window = Tk()
        self.setup_gui()

    def send_message(self):
        user_input = self.entry.get().strip()
        if user_input.lower() == 'exit':
            messagebox.showinfo("Chatbot", "Goodbye!")
            self.window.destroy()
            return
        start_time = time.time()
        # if user_input is not empty string
        if user_input:
            assistant_response = self.chatbot.get_response(user_input)
        else:
            assistant_response = ""
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken by the LLM: {time_taken} seconds")
        self.tts_engine.speak(assistant_response)
        self.display_message(user_input, assistant_response)
        self.entry.delete(0, END)

    def display_message(self, user_input, assistant_response):
        self.chat_history.insert(END, f'User: {user_input}\n Coach: {assistant_response}\n\n')
        self.chat_history.see(END)
        self.chatbot.record_data(user_input, assistant_response)
        

    def clear_chat(self):
        self.chat_history.delete('1.0', END)

    def setup_gui(self):
        self.window.title(" Coach GUI")
        self.window.configure(bg="#f0f0f0")

        header_label = Label(self.window, text="AI Golf Coach", font=("Lato", 20, "bold"), bg="#f0f0f0")
        header_label.pack(side=tk.TOP, padx=10, pady=(20, 10))
        
        separator = Frame(self.window, height=2, width=600, bg="#a6a6a6")
        separator.pack(pady=(20, 10))

        self.chat_history = scrolledtext.ScrolledText(self.window, wrap=WORD, width=60, height=20, font=("Lato", 16), bg="#ffffff")
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=20, side=tk.LEFT)

        # Load and display the image
        image_path = 'C:\\Users\\berkg\\source\\AI_Coach\\racoon.png'
        original_image = Image.open(image_path)
        resized_image = original_image.resize((370, 450), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(resized_image)
        image_label = Label(self.window, image=image, bg="#f0f0f0")
        image_label.pack(side=tk.TOP, padx=5, pady=(5, 0))
        image_label.image = image

        clean_chat_button = Button(self.window, text="Clean Chat", command=self.clear_chat, font=("Roboto", 18), bg="green", fg="#ffffff", activebackground="#0056b3", activeforeground="#ffffff")
        clean_chat_button.pack(side=tk.TOP, padx=10, pady=(10, 20))

        send_button = Button(self.window, text="Send", command=self.send_message, font=("Roboto", 18), bg="#007bff", fg="#ffffff", activebackground="#0056b3", activeforeground="#ffffff", height=2, width=10)
        send_button.pack(side=tk.BOTTOM, padx=10, pady=(10, 30))

        self.entry = Entry(self.window, width=60, font=("Roboto", 16))
        self.entry.pack(side=tk.BOTTOM, padx=10, pady=(10, 20))

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    model_path = "C:\\Users\\berkg\\source\\AI_Coach\\zephyr_beta_golf_ggml-model-q4.gguf"
    chatbot = Chatbot(model_path)
    tts_engine = TextToSpeech()
    gui = ChatbotGUI(chatbot, tts_engine)
    gui.run()
