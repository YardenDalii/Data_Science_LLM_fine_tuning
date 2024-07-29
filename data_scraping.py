
import pymongo as pm
import certifi
import requests
import wikipedia
import json
import re
import os
from dotenv import load_dotenv

load_dotenv() 

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



MONGODB_URI = os.getenv("MONGODB_URI")

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")

# def database_connection(uri):

#     client = pm.MongoClient(uri, tlsCAFile=certifi.where())
#     # client = pm.MongoClient(uri)
#     db = client.wikipedia_data
#     collection = db.articles

#     return collection

def database_connection(uri):
    try:
        client = pm.MongoClient(
            uri,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Try this if certifi.where() does not solve the problem
            tlsCAFile=certifi.where()
        )
        db = client.wikipedia_data
        collection = db.articles
        logger.info("Successfully connected to MongoDB")
        return collection
    except pm.errors.ServerSelectionTimeoutError as err:
        logger.error(f"Failed to connect to MongoDB: {err}")
        raise


# Fetch Wikipedia data
# def fetch_wikipedia_data(query, max_results=10, links=5):
#     search_results = wikipedia.search(query, results=max_results)
    
#     data = {}
#     for result in search_results:
#         try:
#             page = wikipedia.page(result)
#             data[page.url] = {"title": page.title, "content": page.content, "links": page.links[:links]}
#         except wikipedia.exceptions.DisambiguationError as e:
#             print(f"Can't read page: {page.title}")
#             continue
#         except wikipedia.exceptions.PageError as e:
#             print(f"Can't read page: {page.title}")
#             continue

#     return data

def fetch_wikipedia_data(query, max_results=10, links=5):
    search_results = wikipedia.search(query, results=max_results)
    
    data = {}
    for result in search_results:
        try:
            page = wikipedia.page(result)
            data[page.url] = {"title": page.title, "content": page.content, "links": page.links[:links]}
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"DisambiguationError for page: {result}")
            continue
        except wikipedia.exceptions.PageError as e:
            logger.warning(f"PageError for page: {result}")
            continue

    return data


# Clean text data
def clean_text(text):
    text = re.sub(r'\[.*?\]+', '', text)  # Remove references
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text



def generate_embeddings(text: str):
    response = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}.")
    
    return response.json()



def get_answer(query: str):

    collection = database_connection(MONGODB_URI)
    results = collection.aggregate([
        { "$vectorSearch": {
            "queryVector": generate_embeddings(query),
            "path": "embedding",
            "numCandidates": 100,
            "limit": 5,
            "index": "default",
        }}
    ])

    return results


# Main function to execute the workflow
# def main():
#     # Connect to MongoDB
#     collection = database_connection(os.getenv("MONGODB_URI"))
    
#     # Query Wikipedia
#     queries = ["Israeli-Arab conflict", "Six-Day War", "Yom Kippur War", "Israel's Independance War", "Israel-Palestine conflict, October 7th war"]
#     raw_data = fetch_wikipedia_data(queries, links=20)
    
#     # Process and store data in MongoDB
#     for url, content in raw_data.items():
#         cleaned_content = clean_text(content['content'])
#         embedding = generate_embeddings(cleaned_content)
        
#         document = {
#             "url": url,
#             "title": content['title'],
#             "content": cleaned_content,
#             "links": content["links"],
#             "embedding": embedding
#         }
        
#         collection.insert_one(document)
#         print(f"Inserted document for {content['title']}")
#     # query = "What started the 'six-days war'?"
#     # results = get_answer(query)

#     # for r in results:
#     #     print(f"Title: {r["title"]},\nText: {r["content"]}\n")

# if __name__ == "__main__":
#     main()


def main():
    try:
        # Connect to MongoDB
        collection = database_connection(MONGODB_URI)
        
        # Query Wikipedia
        queries = ["Israeli-Arab conflict", "Six-Day War", "Yom Kippur War", "Israel's Independence War", "Israel-Palestine conflict, October 7th war"]
        raw_data = fetch_wikipedia_data(queries)
        
        # Process and store data in MongoDB
        for url, content in raw_data.items():
            cleaned_content = clean_text(content['content'])
            embedding = generate_embeddings(cleaned_content)
            
            document = {
                "url": url,
                "title": content['title'],
                "content": cleaned_content,
                "links": content['links'],
                "embedding": embedding
            }
            
            collection.update_one(
                {"url": url},  # Filter
                {"$set": document},  # Update operation
                upsert=True  # Insert the document if it does not exist
            )
            logger.info(f"Inserted document for {content['title']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()