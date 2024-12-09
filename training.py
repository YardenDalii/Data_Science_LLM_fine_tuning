from data_scraping import cleaning, save_as_json, fetch_and_process_linked_pages, fetch_wikipedia_data, prepare_training_data
from data_scraping import DATA_DIR, TRAIN_FILE_PATH, HF_TOKEN
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, AutoConfig
import torch

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_data():
    queries = [
    "Arab–Israeli conflict",
    "Israeli–Palestinian conflict",
    "Israeli Declaration of Independence",
    "Six-Day War",
    "Gulf War",
    "Israel–Hamas war",
    "2006 Lebanon War",
    "1982 Lebanon War",
    "2024 Iran–Israel conflict",
    "Iran–Israel proxy conflict",
    "Israel–Hezbollah conflict",
    "Yom Kippur War",
    "2014 Gaza War",
    "History of Israel",
    "History of Palestine",
    "2023 Hamas-led attack on Israel",
    ]

    fetch_wikipedia_data(queries)

    logger.info("All documents have been stored successfully!")

# store_data()



def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )


def load_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

# Training function
def train(train_file_path, model_name, output_dir, overwrite_output_dir,per_device_train_batch_size, num_train_epochs, save_steps, learning_rate):
    
    token = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True if device == "cuda" else False,  # Enable fp16 for CUDA
        #evaluation_strategy="steps",
        save_total_limit=2,
        learning_rate=learning_rate,
        no_cuda=device == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()
    trainer.save_model()


# Training parameters
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./fine_tuned_tinyllama_v0.1"
# model_name = "gpt2"
# output_dir = "./fine_tuned_gpt2_v0.1"
overwrite_output_dir = True
per_device_train_batch_size = 8
num_train_epochs = 6
save_steps = 1500
learning_rate = 5e-5


# Prepare training data
prepare_training_data(DATA_DIR, TRAIN_FILE_PATH)


train(
    train_file_path=TRAIN_FILE_PATH,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    learning_rate=learning_rate
)