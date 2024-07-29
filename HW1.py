import pymongo as pm
import wikipedia
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext
import praw
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")


def database_connection(uri):
    client = pm.MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
    db = client.hw_1
    collection = db.articles
    return collection


# Fetch Wikipedia data
def fetch_wikipedia_data(query, max_results=2):
    search_results = wikipedia.search(query, results=max_results)
    
    data = {}
    for result in search_results:
        try:
            page = wikipedia.page(result)
            data[page.url] = {"title": page.title, "content": page.content, "source": "Wikipedia"}
        except wikipedia.exceptions.DisambiguationError:
            continue
        except wikipedia.exceptions.PageError:
            continue

    return data


# Fetch Reddit data
def fetch_reddit_data(query):
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    try:
        subreddit = reddit.subreddit(query)
        subreddit_exists = subreddit.display_name
    except Exception as e:
        return None, str(e)
    
    data = {}
    for submission in subreddit.hot(limit=10):
        data[submission.url] = {"title": submission.title, "content": submission.selftext, "source": "Reddit"}
    return data, None


# Clean text data
def clean_text(text):
    text = re.sub(r'\[.*?\]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# Store data in MongoDB
def store_in_mongodb(data, collection):
    for url, content in data.items():
        collection.update_one({"url": url}, {"$set": content}, upsert=True)


# Retrieve data from MongoDB
def retrieve_from_mongodb(collection, source):
    data = collection.find({"source": source})
    return data


# Tkinter GUI
def scrape_and_store():
    print("pressed!")
    query = entry.get()
    if query:
        source = source_var.get()
        if source == "Wikipedia":
            data = fetch_wikipedia_data(query)
            store_in_mongodb(data, collection)
            messagebox.showinfo("Success", f"Data for '{query}' from {source} has been stored in MongoDB.")
        elif source == "Reddit":
            data, error = fetch_reddit_data(query)
            if error:
                messagebox.showwarning("Error", f"Could not fetch data from Reddit: {error}")
            else:
                store_in_mongodb(data, collection)
                messagebox.showinfo("Success", f"Data for '{query}' from {source} has been stored in MongoDB.")
    else:
        messagebox.showwarning("Input Error", "Please enter a search query.")

def retrieve_data():
    print("pressed!")
    source = source_var.get()
    collection = database_connection(MONGODB_URI)
    data = retrieve_from_mongodb(collection, source)
    display_text.delete('1.0', tk.END)
    for record in data:
        display_text.insert(tk.END, f"Source: {record['source']}\nURL: {record['url']}\nTitle: {record['title']}\nContent: {record['content']}\n\n")

collection = database_connection(MONGODB_URI)

app = tk.Tk()
app.title("Data Scraper")

tk.Label(app, text="Select Source:").pack(pady=5)
source_var = tk.StringVar(value="Wikipedia")
tk.Radiobutton(app, text="Wikipedia", variable=source_var, value="Wikipedia").pack(anchor=tk.W)
tk.Radiobutton(app, text="Reddit", variable=source_var, value="Reddit").pack(anchor=tk.W)

tk.Label(app, text="Enter Search Query:").pack(pady=5)
entry = tk.Entry(app, width=40)
entry.pack(pady=5)

tk.Button(app, text="Scrape and Store", command=scrape_and_store).pack(pady=10)
tk.Button(app, text="Retrieve Data", command=retrieve_data).pack(pady=10)

display_text = scrolledtext.ScrolledText(app, width=80, height=20)
display_text.pack(pady=10)

app.mainloop()