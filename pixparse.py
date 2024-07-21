"""
    Based on the dataset "pixparse/pdfa-eng-wds" from Hugging Face Datasets.
    https://huggingface.co/datasets/pixparse/pdfa-eng-wds
    keys=dict_keys(['__key__', '__url__', 'json', 'pdf'])
"""
import json
import os
from datasets import load_dataset

NUM_SAMPLES = 20
DIR_NAME = "pdfa-eng-wds"

def save_data(filename, data):
    with open(filename, "wb") as f:
        f.write(data)

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

dataset = load_dataset("pixparse/pdfa-eng-wds", streaming=True)
train = dataset["train"]
train_iter = iter(train)
os.makedirs(DIR_NAME, exist_ok=True)

for i in range(NUM_SAMPLES):
    row = next(train_iter )
    key = row["__key__"]
    pdf = row["pdf"]
    meta = row["json"]
    pdf_path = os.path.join(DIR_NAME, f"{key}.pdf")
    json_path = os.path.join(DIR_NAME, f"{key}.json")
    save_data(pdf_path, pdf)
    save_json(json_path, meta)
    print(f"{i:4}: saved {pdf_path} with {len(pdf)} bytes")
