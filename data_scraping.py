from transformers import pipeline
import wikipedia
import json
import os
import pandas as pd
import numpy as np
import re
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_DIR = "./wikipedia_data"

TRAIN_FILE_PATH = "train_data.txt"

HF_TOKEN = "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"


def cleaning(s):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    return s


def save_as_json(data, file_path):
    """Save the provided data as a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved data to {file_path}")


def fetch_wikipedia_data(queries, max_results=10, links=12):
    """Fetch data from Wikipedia based on the given queries and save as JSON files."""
    for query in queries:
        search_results = wikipedia.search(query, results=max_results)
        for result in search_results:
            try:
                page = wikipedia.page(result)
                file_path = f"{DATA_DIR}/{page.title}.json"

                # Check for duplicates
                if os.path.exists(file_path):
                    print(f"Document for {page.title} already exists, skipping.")
                    continue

                data = {
                    "url": page.url,
                    "title": page.title,
                    "content": cleaning(page.content),
                    "links": page.links[:links],
                    "parent_url": None  # No parent URL for main pages
                }

                save_as_json(data, file_path)

                fetch_and_process_linked_pages(data['links'], page.url)

            except wikipedia.exceptions.DisambiguationError:
                logger.error(f"DisambiguationError for page: {result}")
                continue

            except wikipedia.exceptions.PageError:
                logger.error(f"PageError for page: {result}")
                continue


def fetch_and_process_linked_pages(links, parent_url):
    """Fetch and process the pages linked from the main page and save as JSON files."""
    for link in links:
        try:
            linked_page = wikipedia.page(link)
            file_path = f"{DATA_DIR}/{linked_page.title}.json"

            # Check for duplicates
            if os.path.exists(file_path):
                print(f"Document for {linked_page.title} already exists, skipping.")
                continue

            cleaned_content = cleaning(linked_page.content)

            document = {
                "url": linked_page.url,
                "title": linked_page.title,
                "content": cleaned_content,
                "links": linked_page.links[:5],  # Fetch top 5 links from the linked page
                "parent_url": parent_url  # Keep track of the original page
            }

            save_as_json(document, file_path)

        except wikipedia.exceptions.DisambiguationError:
            logger.error(f"DisambiguationError for linked page: {link}")
            continue
        except wikipedia.exceptions.PageError:
            logger.error(f"PageError for linked page: {link}")
            continue




def prepare_training_data(data_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    outfile.write(data['content'] + "\n")
                    logger.info(f"Added content from {file_name} to training data.")