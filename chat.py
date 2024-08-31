from transformers import set_seed, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import tkinter as tk
from tkinter import ttk
from transformers import pipeline
import time


GPT_PATH = "./fine_tuned_gpt2_v0.1"

LLAMA_PATH = "./fine_tuned_tinyLLaMa_v0.1/checkpoint"


set_seed(42)

gpt2_model = GPT2LMHeadModel.from_pretrained(GPT_PATH, use_safetensors=True)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_generator = pipeline('text-generation', model=gpt2_model, tokenizer=gpt2_tokenizer)

tinyllama_model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH)
tinyllama_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tinyLLaMa_v0.1")
tinyllama_generator = pipeline('text-generation', model=tinyllama_model, tokenizer=tinyllama_tokenizer)


def type_text(text_widget, text, delay=10):
    """Type text character by character in the text_widget with a delay."""
    for char in text:
        text_widget.insert(tk.END, char)
        text_widget.update()
        time.sleep(delay / 12000.0)  # Convert milliseconds to seconds

def get_model_responses():
    # Get the user input from the text field
    question = question_entry.get()
    # what is the israel-palestine conflict all about?
    
    # Clear previous results
    result_text.delete(1.0, tk.END)

    # Get the responses from both models
    tiny_llama_response = tinyllama_generator(question,
                                              max_length=250,
                                              num_return_sequences=1,
                                                temperature=0.2,
                                                top_p=0.9,
                                                top_k=50,
                                                repetition_penalty=1.2,
                                                truncation=True)[0]['generated_text']
    
    gpt2_response = gpt2_generator(question,
                                   max_length=250,
                                   num_return_sequences=1,
                                    temperature=0.2,
                                    top_p=0.9,
                                    top_k=50,
                                    repetition_penalty=1.2,
                                    truncation=True)[0]['generated_text']

    # Display the responses with typing animation
    result_text.insert(tk.END, "TinyLLaMa Response:\n")
    type_text(result_text, tiny_llama_response + "\n\n", delay=50)
    result_text.insert(tk.END, "GPT-2 Response:\n")
    type_text(result_text, gpt2_response + "\n\n", delay=50)

# Create the main window
root = tk.Tk()
root.title("Model Comparison")

# Create a label and text entry for the user input
question_label = tk.Label(root, text="Enter your question:")
question_label.pack(pady=10)

question_entry = tk.Entry(root, width=50)
question_entry.pack(pady=10)

# Create a button to trigger the comparison
compare_button = tk.Button(root, text="Compare Models", command=get_model_responses)
compare_button.pack(pady=10)

# Create a Text widget for displaying the results
result_text = tk.Text(root, wrap=tk.WORD, height=20, width=80)
result_text.pack(pady=20)

# Run the application
root.mainloop()