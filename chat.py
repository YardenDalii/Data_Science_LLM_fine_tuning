
# what is the israel-palestine conflict is all about?
# What are the events that happened following the Yom Kippur War?

import tkinter as tk
from tkinter import ttk
from transformers import set_seed, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np
import os
import json

GPT_PATH = "./fine_tuned_gpt2_v0.1" # Link to drive folder: https://drive.google.com/drive/folders/1F-5xuW0XL1sh9PFZEekjLDxcCnzPic15?usp=share_link
LLAMA_PATH = "./fine_tuned_tinyLLaMa_v0.1/checkpoint" # Link to drive folder: https://drive.google.com/drive/folders/1-JgC1vxdbKUX0zdQA2f0n-skl6hKc2dz?usp=share_link
DATA_DIR = "./wikipedia_data"

set_seed(42)

gpt2_model = GPT2LMHeadModel.from_pretrained(GPT_PATH, use_safetensors=True)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_generator = pipeline('text-generation', model=gpt2_model, tokenizer=gpt2_tokenizer, device=0)

tinyllama_model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH)
tinyllama_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tinyLLaMa_v0.1")
tinyllama_generator = pipeline('text-generation', model=tinyllama_model, tokenizer=tinyllama_tokenizer, device=0)

def type_text(text_widget, text, delay=10):
    for char in text:
        text_widget.insert(tk.END, char)
        text_widget.update()

def find_relevant_documents(query, k=3):
    documents = []
    file_names = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)  # Parse JSON
                    content = data.get("content", "").strip()
                    if content:
                        documents.append(content)
                        file_names.append(filename)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
 
    if not documents:
        print("No documents found or all documents are empty.")
        return []
    

    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    

    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    relevant_docs = [(documents[index], similarities[index], file_names[index]) for index in top_k_indices]
    return relevant_docs


def calculate_metrics(response, relevant_docs):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    metrics_results = []
    
    for doc, _, _ in relevant_docs:
        # ROUGE-L score
        rouge_score = rouge.score(response, doc)['rougeL'].fmeasure
        
        # Keyword matching score
        doc_keywords = set(doc.split())
        response_keywords = set(response.split())
        common_keywords = doc_keywords.intersection(response_keywords)
        keywords_score = len(common_keywords) / len(response_keywords) if response_keywords else 0
        
        # Cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        doc_vectors = vectorizer.fit_transform([doc, response])
        cosine_score = cosine_similarity(doc_vectors[0:1], doc_vectors[1:2]).flatten()[0]
        
        metrics_results.append((rouge_score, keywords_score, cosine_score))
    
    return metrics_results

def display_metrics(model_name, metrics_results):
    avg_rouge = np.mean([metrics[0] for metrics in metrics_results])
    avg_keywords = np.mean([metrics[1] for metrics in metrics_results])
    avg_cosine = np.mean([metrics[2] for metrics in metrics_results])
    
    result_text.insert(tk.END, f"Average {model_name} Metrics across top documents:\n")
    result_text.insert(tk.END, f"ROUGE-L Score: {avg_rouge:.2f}\n")
    result_text.insert(tk.END, f"Keyword Matching Score: {avg_keywords:.2f}\n")
    result_text.insert(tk.END, f"Cosine Similarity Score: {avg_cosine:.2f}\n\n")

def compare_models(tiny_llama_metrics, gpt2_metrics):
    tiny_llama_wins = 0
    gpt2_wins = 0
    
    for i in range(len(tiny_llama_metrics)):
        # Compare ROUGE-L scores
        if tiny_llama_metrics[i][0] > gpt2_metrics[i][0]:
            tiny_llama_wins += 1
        elif gpt2_metrics[i][0] > tiny_llama_metrics[i][0]:
            gpt2_wins += 1
        
        # Compare Keyword Matching scores
        if tiny_llama_metrics[i][1] > gpt2_metrics[i][1]:
            tiny_llama_wins += 1
        elif gpt2_metrics[i][1] > tiny_llama_metrics[i][1]:
            gpt2_wins += 1
        
        # Compare Cosine Similarity scores
        if tiny_llama_metrics[i][2] > gpt2_metrics[i][2]:
            tiny_llama_wins += 1
        elif gpt2_metrics[i][2] > tiny_llama_metrics[i][2]:
            gpt2_wins += 1
    
    better_model = "TinyLLaMa" if tiny_llama_wins > gpt2_wins else "GPT-2"
    result_text.insert(tk.END, f"The model that performed better overall is: {better_model}\n")
    result_text.insert(tk.END, f"TinyLLaMa wins: {tiny_llama_wins}, GPT-2 wins: {gpt2_wins}\n")

def get_model_responses():
    question = question_entry.get()
    result_text.delete(1.0, tk.END)
    
    progress_bar['value'] = 0
    root.update_idletasks()
    
    progress_bar['value'] = 20
    root.update_idletasks()
    relevant_docs = find_relevant_documents(question, k=3)
    
    progress_bar['value'] = 40
    root.update_idletasks()
    try:
        tiny_llama_response = tinyllama_generator(
            question,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True
        )[0]['generated_text']
    except IndexError as e:
        tiny_llama_response = f"Error generating response: {str(e)}"
        print(f"Error with TinyLLaMa: {e}")
    
    progress_bar['value'] = 60
    root.update_idletasks()
    try:
        gpt2_response = gpt2_generator(
            question,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.6,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True
        )[0]['generated_text']
    except IndexError as e:
        gpt2_response = f"Error generating response: {str(e)}"
        print(f"Error with GPT-2: {e}")
    

    progress_bar['value'] = 80
    root.update_idletasks()
    result_text.insert(tk.END, "Top K Relevant Documents:\n")
    for i, (doc, score, filename) in enumerate(relevant_docs):
        result_text.insert(tk.END, f"Document {i+1} (Score: {score:.2f}, File: {filename}):\n{doc[:200]}...\n\n")
    

    result_text.insert(tk.END, "TinyLLaMa Response:\n")
    type_text(result_text, tiny_llama_response + "\n\n", delay=50)
    tiny_llama_metrics = calculate_metrics(tiny_llama_response, relevant_docs)
    display_metrics("TinyLLaMa", tiny_llama_metrics)
    
    result_text.insert(tk.END, "GPT-2 Response:\n")
    type_text(result_text, gpt2_response + "\n\n", delay=50)
    gpt2_metrics = calculate_metrics(gpt2_response, relevant_docs)
    display_metrics("GPT-2", gpt2_metrics)
    

    result_text.insert(tk.END, "Model Performance Comparison:\n")
    compare_models(tiny_llama_metrics, gpt2_metrics)
    

    progress_bar['value'] = 100
    root.update_idletasks()

# Tkinter setup
root = tk.Tk()
root.title("Model Comparison")

question_label = tk.Label(root, text="Enter your question:")
question_label.pack(pady=10)

question_entry = tk.Entry(root, width=50)
question_entry.pack(pady=10)

compare_button = tk.Button(root, text="Compare Models", command=get_model_responses)
compare_button.pack(pady=10)


progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

result_text = tk.Text(root, wrap=tk.WORD, height=30, width=100)
result_text.pack(pady=20)

root.mainloop()



# import tkinter as tk
# from tkinter import ttk, messagebox
# from transformers import set_seed, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from rouge_score import rouge_scorer
# import numpy as np
# import os
# import json
# import threading
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPT_PATH = "./fine_tuned_gpt2_v0.1"
# LLAMA_PATH = "./fine_tuned_tinyLLaMa_v0.1/checkpoint"
# DATA_DIR = "./wikipedia_data"

# set_seed(42)

# # Load models with error handling
# try:
#     gpt2_model = GPT2LMHeadModel.from_pretrained(GPT_PATH)
#     gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     gpt2_generator = pipeline('text-generation', model=gpt2_model, tokenizer=gpt2_tokenizer)
# except Exception as e:
#     logging.error(f"Error loading GPT-2 model: {e}")
#     gpt2_generator = None

# try:
#     tinyllama_model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH)
#     tinyllama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
#     tinyllama_generator = pipeline('text-generation', model=tinyllama_model, tokenizer=tinyllama_tokenizer)
# except Exception as e:
#     logging.error(f"Error loading TinyLLaMa model: {e}")
#     tinyllama_generator = None

# def type_text(text_widget, text, delay=10):
#     for char in text:
#         text_widget.insert(tk.END, char)
#         text_widget.update_idletasks()
#         text_widget.after(delay)

# def find_relevant_documents(query, k=3):
#     documents = []
#     file_names = []

#     for filename in os.listdir(DATA_DIR):
#         if filename.endswith('.json'):
#             file_path = os.path.join(DATA_DIR, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     content = data.get("content", "").strip()
#                     if content:
#                         documents.append(content)
#                         file_names.append(filename)
#             except json.JSONDecodeError:
#                 logging.error(f"Error decoding JSON in file: {filename}")
#             except Exception as e:
#                 logging.error(f"Error reading file {filename}: {e}")
    
#     if not documents:
#         logging.info("No documents found or all documents are empty.")
#         return []
    
#     vectorizer = TfidfVectorizer(stop_words='english')
#     doc_vectors = vectorizer.fit_transform(documents)
    
#     query_vector = vectorizer.transform([query])
    
#     similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
#     top_k_indices = similarities.argsort()[-k:][::-1]
    
#     relevant_docs = [(documents[index], similarities[index], file_names[index]) for index in top_k_indices]
#     return relevant_docs

# def calculate_metrics(response, relevant_docs):
#     rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
#     metrics_results = []
    
#     for doc, _, _ in relevant_docs:
#         # ROUGE-L score
#         rouge_score = rouge.score(response, doc)['rougeL'].fmeasure
        
#         # Keyword matching score
#         doc_keywords = set(doc.split())
#         response_keywords = set(response.split())
#         common_keywords = doc_keywords.intersection(response_keywords)
#         keywords_score = len(common_keywords) / len(response_keywords) if response_keywords else 0
        
#         # Cosine similarity
#         vectorizer = TfidfVectorizer(stop_words='english')
#         doc_vectors = vectorizer.fit_transform([doc, response])
#         cosine_score = cosine_similarity(doc_vectors[0:1], doc_vectors[1:2]).flatten()[0]
        
#         metrics_results.append((rouge_score, keywords_score, cosine_score))
    
#     return metrics_results

# def display_metrics(model_name, metrics_results, result_text):
#     avg_rouge = np.mean([metrics[0] for metrics in metrics_results])
#     avg_keywords = np.mean([metrics[1] for metrics in metrics_results])
#     avg_cosine = np.mean([metrics[2] for metrics in metrics_results])
    
#     result_text.insert(tk.END, f"Average {model_name} Metrics across top documents:\n")
#     result_text.insert(tk.END, f"ROUGE-L Score: {avg_rouge:.2f}\n")
#     result_text.insert(tk.END, f"Keyword Matching Score: {avg_keywords:.2f}\n")
#     result_text.insert(tk.END, f"Cosine Similarity Score: {avg_cosine:.2f}\n\n")

# def compare_models(tiny_llama_metrics, gpt2_metrics, result_text):
#     tiny_llama_wins = 0
#     gpt2_wins = 0
    
#     for i in range(len(tiny_llama_metrics)):
#         # Compare ROUGE-L scores
#         if tiny_llama_metrics[i][0] > gpt2_metrics[i][0]:
#             tiny_llama_wins += 1
#         elif gpt2_metrics[i][0] > tiny_llama_metrics[i][0]:
#             gpt2_wins += 1
        
#         # Compare Keyword Matching scores
#         if tiny_llama_metrics[i][1] > gpt2_metrics[i][1]:
#             tiny_llama_wins += 1
#         elif gpt2_metrics[i][1] > tiny_llama_metrics[i][1]:
#             gpt2_wins += 1
        
#         # Compare Cosine Similarity scores
#         if tiny_llama_metrics[i][2] > gpt2_metrics[i][2]:
#             tiny_llama_wins += 1
#         elif gpt2_metrics[i][2] > tiny_llama_metrics[i][2]:
#             gpt2_wins += 1
    
#     better_model = "TinyLLaMa" if tiny_llama_wins > gpt2_wins else "GPT-2"
#     result_text.insert(tk.END, f"The model that performed better overall is: {better_model}\n")
#     result_text.insert(tk.END, f"TinyLLaMa wins: {tiny_llama_wins}, GPT-2 wins: {gpt2_wins}\n")

# def show_results_window(question, tiny_llama_response, gpt2_response, relevant_docs, tiny_llama_metrics, gpt2_metrics):
#     result_window = tk.Toplevel(root)
#     result_window.title("Model Comparison Results")
#     result_text = tk.Text(result_window, wrap=tk.WORD, height=30, width=100)
#     result_text.pack(pady=20)
    
#     result_text.insert(tk.END, "Top K Relevant Documents:\n")
#     for i, (doc, score, filename) in enumerate(relevant_docs):
#         result_text.insert(tk.END, f"Document {i+1} (Score: {score:.2f}, File: {filename}):\n{doc[:200]}...\n\n")
    
#     result_text.insert(tk.END, "TinyLLaMa Response:\n")
#     type_text(result_text, tiny_llama_response + "\n\n", delay=50)
#     display_metrics("TinyLLaMa", tiny_llama_metrics, result_text)
    
#     result_text.insert(tk.END, "GPT-2 Response:\n")
#     type_text(result_text, gpt2_response + "\n\n", delay=50)
#     display_metrics("GPT-2", gpt2_metrics, result_text)
    
#     result_text.insert(tk.END, "Model Performance Comparison:\n")
#     compare_models(tiny_llama_metrics, gpt2_metrics, result_text)

# def get_model_responses():
#     question = question_entry.get()
#     if not question:
#         messagebox.showwarning("Input Error", "Please enter a question.")
#         return
    
#     progress_bar['value'] = 0
#     root.update_idletasks()
    
#     progress_bar['value'] = 20
#     root.update_idletasks()
#     relevant_docs = find_relevant_documents(question, k=3)
    
#     progress_bar['value'] = 40
#     root.update_idletasks()
#     try:
#         tiny_llama_response = tinyllama_generator(
#             question,
#             max_new_tokens=150,
#             num_return_sequences=1,
#             temperature=0.6,
#             top_p=0.8,
#             top_k=50,
#             repetition_penalty=1.2,
#             do_sample=True
#         )[0]['generated_text']
#     except Exception as e:
#         tiny_llama_response = f"Error generating response: {str(e)}"
#         logging.error(f"Error with TinyLLaMa: {e}")
    
#     progress_bar['value'] = 60
#     root.update_idletasks()
#     try:
#         gpt2_response = gpt2_generator(
#             question,
#             max_new_tokens=150,
#             num_return_sequences=1,
#             temperature=0.6,
#             top_p=0.8,
#             top_k=50,
#             repetition_penalty=1.2,
#             do_sample=True
#         )[0]['generated_text']
#     except Exception as e:
#         gpt2_response = f"Error generating response: {str(e)}"
#         logging.error(f"Error with GPT-2: {e}")
    
#     progress_bar['value'] = 80
#     root.update_idletasks()
    
#     tiny_llama_metrics = calculate_metrics(tiny_llama_response, relevant_docs)
#     gpt2_metrics = calculate_metrics(gpt2_response, relevant_docs)
    
#     progress_bar['value'] = 100
#     root.update_idletasks()
    
#     # Open the result window
#     show_results_window(question, tiny_llama_response, gpt2_response, relevant_docs, tiny_llama_metrics, gpt2_metrics)

# # Tkinter setup
# root = tk.Tk()
# root.title("Model Comparison")

# question_label = tk.Label(root, text="Enter your question:")
# question_label.pack(pady=10)

# question_entry = tk.Entry(root, width=50)
# question_entry.pack(pady=10)

# compare_button = tk.Button(root, text="Compare Models", command=lambda: threading.Thread(target=get_model_responses).start())
# compare_button.pack(pady=10)

# progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
# progress_bar.pack(pady=10)

# root.mainloop()