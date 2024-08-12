
import pymongo as pm
import certifi
import requests
import wikipedia
import json
import re
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document


MONGODB_URI = "mongodb+srv://yardeda2:yardeda2@dataproject.zf7is.mongodb.net/?retryWrites=true&w=majority&appName=DataProject"

MODEL_NAME = "gpt2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

HF_TOKEN = "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"
EMBEDDING_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def database_connection(uri):
    client = pm.MongoClient(uri, tlsCAFile=certifi.where())
    db = client.wikipedia
    collection = db.data
    return collection


def fetch_wikipedia_data(queries, max_results=10, links=5):
    """Fetch data from Wikipedia based on the given queries."""
    data = {}
    for query in queries:
        search_results = wikipedia.search(query, results=max_results)
        for result in search_results:
            try:
                page = wikipedia.page(result)
                data[page.url] = {
                    "title": page.title,
                    "content": page.content,
                    "links": page.links[:links]
                }
            except wikipedia.exceptions.DisambiguationError:
                logger.warning(f"DisambiguationError for page: {result}")
                continue
            except wikipedia.exceptions.PageError:
                logger.warning(f"PageError for page: {result}")
                continue
    return data



def clean_text(text):
    text = re.sub(r'\[.*?\]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text



def fetch_and_process_linked_pages(collection, links, parent_url, embeddings: Embeddings):
    """Fetch and process the pages linked from the main page."""
    for link in links:
        try:
            linked_page = wikipedia.page(link)
            
            if document_exists(collection, linked_page.title):
                logger.info(f"Document already exists for {linked_page.title}, skipping.")
                continue
            
            cleaned_content = clean_text(linked_page.content)
            embedding = embeddings.embed_documents([cleaned_content])[0]

            document = {
                "url": linked_page.url,
                "title": linked_page.title,
                "content": cleaned_content,
                "links": linked_page.links[:5],  # Fetch top 5 links from the linked page
                "embedding": embedding,
                "parent_url": parent_url  # Keep track of the original page
            }

            collection.insert_one(document)
            logger.info(f"Inserted linked document for {linked_page.title} under parent {parent_url}")
        
        except wikipedia.exceptions.DisambiguationError:
            logger.warning(f"DisambiguationError for linked page: {link}")
            continue
        except wikipedia.exceptions.PageError:
            logger.warning(f"PageError for linked page: {link}")
            continue



def process_and_store_data(collection, raw_data, embeddings: Embeddings):
    """Process the raw Wikipedia data and store it in MongoDB."""
    for url, content in raw_data.items():
        if document_exists(collection, content['title']):
            logger.info(f"Document already exists for {content['title']}, skipping.")
            continue

        cleaned_content = clean_text(content['content'])
        embedding = embeddings.embed_documents([cleaned_content])[0]
        
        document = {
            "url": url,
            "title": content['title'],
            "content": cleaned_content,
            "links": content['links'],
            "embedding": embedding
        }

        collection.insert_one(document)
        logger.info(f"Inserted document for {content['title']}")

        fetch_and_process_linked_pages(collection, content['links'], url, embeddings)



# def ask_model(query: str, embeddings: Embeddings, top_k: int = 3):

#     collection = database_connection(MONGODB_URI)
    
#     # Step 2: Generate embeddings for the query
#     query_embedding = embeddings.embed_query(query)
    
#     # Step 3: Search for the most relevant documents using vector search
#     results = collection.aggregate([
#         {
#             "$vectorSearch": {
#                 "queryVector": query_embedding,
#                 "path": "embedding",
#                 "numCandidates": 100,
#                 "limit": top_k,
#                 "index": "WikipediaDataSearch",
#             }
#         }
#     ])
    
#     # Step 4: Construct the response based on the top-k documents
#     response = "Based on the information we have:\n\n"
#     response_dict = { "title": [], "content": []}
#     for i, result in enumerate(results):
#         title = result['title']
#         content = result['content']#[:500]  # Limiting content to 500 characters for brevity
#         response_dict["title"].append(title)
#         response_dict["content"].append(content)
#         response += f"{i+1}. **{title}**: {content}...\n\n"
    
#     if not response.strip():
#         response = "Sorry, I couldn't find relevant information in the database."
    
#     return response, response_dict

def ask_model(query: str, embeddings: Embeddings, top_k: int = 3):

    collection = database_connection(MONGODB_URI)
    
    query_embedding = embeddings.embed_query(query)
    
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": top_k,
                "index": "WikipediaDataSearch",
            }
        }
    ])
    
    documents = []
    for result in results:
        title = result.get('title', 'Untitled')
        content = result.get('content', 'No content available')
        metadata = {"title": title}
        documents.append(Document(page_content=content, metadata=metadata))
    
    if not documents:
        response = "Sorry, I couldn't find relevant information in the database."
    else:
        response = "Based on the information we have:\n\n"
        for i, doc in enumerate(documents):
            response += f"{i+1}. **{doc.metadata['title']}**: {doc.page_content[:500]}...\n\n"
    
    return response, documents


def generate_embeddings(text: str):
    response = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}.")
    
    return response.json()

# def ask_model(query: str, top_k=3):
#     collection = database_connection(MONGODB_URI)
    
#     query_embedding = generate_embeddings(query)
    
#     results = collection.aggregate([
#         {
#             "$vectorSearch": {
#                 "queryVector": query_embedding,
#                 "path": "embedding",
#                 "numCandidates": 100,
#                 "limit": top_k,
#                 "index": "WikipediaDataSearch",
#             }
#         }
#     ])
    
#     # Step 4: Construct the response based on the top-k documents
#     response = "Based on the information we have:\n\n"
#     for i, result in enumerate(results):
#         title = result['title']
#         content = result['content'][:500]  # Limiting content to 500 characters for brevity
#         response += f"{i+1}. **{title}**: {content}...\n\n"
    
#     if not response.strip():
#         response = "Sorry, I couldn't find relevant information in the database."
    
#     return response


def document_exists(collection, title):
    """Check if a document with the given title already exists in the collection."""
    return collection.find_one({"title": title}) is not None




def main():
    try:
        # Connect to MongoDB
        collection = database_connection(MONGODB_URI)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # # Query Wikipedia
        # queries = [
        #     "Israeli-Arab conflict", 
        #     "October 7th war", 
        #     "Yom Kippur War", 
        #     "Israel's Independence War", 
        #     "Israel-Palestine conflict",
        # ]
        # raw_data = fetch_wikipedia_data(queries, links=10)
        
        # # Process and store data in MongoDB
        # process_and_store_data(collection, raw_data)

        # Example usage of ask_model
        user_query = "What were the main events during the october 7th masecure?"
        answer = ask_model(user_query, embeddings=embeddings)
        print(answer)
        
        # The rest of your original main function goes here...
        # Connect to MongoDB, fetch Wikipedia data, process and store, etc.
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4