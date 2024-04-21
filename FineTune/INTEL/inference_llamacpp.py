from llama_cpp import Llama
from tkinter import scrolledtext, messagebox
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import re

path = ''
llm = Llama(model_path=path)


def chatbot():
    def send_message():
        user_input = entry.get().strip()

        if user_input.lower() == 'exit':
            messagebox.showinfo("Chatbot", "Goodbye!")
            window.destroy()
            return

        old_text = f"""<|system|>You are an AI assistant. Your job is to help your user specifically on golf exercises and training plans to improve their  skills.
                <user>:{user_input}
                <assistant>:"""
        
        output = llm(old_text, max_tokens=120,
            temperature=0.6,
            top_p=0.5,
            echo=False, 
            stop="<user>:"
            )
        
        output = output["choices"][0]["text"]
        #response_text = output[len(old_text):]
        #extracted_response = response_text.split("<assistant>")[0].strip()
        #print(extracted_response)
        #sentences = re.findall(r'[^.!?]*[.!?]', extracted_response)

        # Joining the sentences to form the trimmed response
        #trimmed_response = ' '.join(sentences).strip()
        #assistant_data = trimmed_response
        assistant_data = output

        chat_history.insert(tk.END, 'User :  ' + user_input + '\n')
        chat_history.insert(tk.END, ' Coach :  ' + assistant_data + '\n\n')
        chat_history.see(tk.END)

        chat_history.tag_configure("user", font=("Roboto", 14, "bold"))
        
        chat_history.tag_configure("assistant", font=("Roboto", 14, "bold"))
        #record_data(user_input, assistant_data)
        entry.delete(0, tk.END)

    def clear_chat():
        chat_history.delete('1.0', tk.END)

    # Create the main window
    window = tk.Tk()
    window.title(" Coach GUI")
    window.configure(bg="#f0f0f0")

    original_image = Image.open('C:\\Users\\berkg\\source\\AI_Coach\\racoon.png') 
    resized_image = original_image.resize((370, 450), Image.Resampling.LANCZOS) 
    image = ImageTk.PhotoImage(resized_image)


    header_label = tk.Label(window, text="AI Golf Coach", font=("Lato", 20, "bold"), bg="#f0f0f0")
    header_label.pack(side=tk.TOP, padx=10, pady=(20, 10))
    separator = tk.Frame(window, height=2, width=600, bg="#a6a6a6")
    separator.pack(pady=(20, 10))

    chat_history = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20,
                                            font=("Lato", 16), bg="#ffffff")
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
    entry = tk.Entry(window, width=60, font=("Roboto", 16))
    entry.pack(side=tk.BOTTOM, padx=10, pady=(10, 20))

    
    window.mainloop()

chatbot()


if __name__ == "__main__":
    chatbot()