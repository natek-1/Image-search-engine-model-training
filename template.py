import os
from pathlib import Path
from src.logger import logging



list_of_files =  [
    "data/embeddings",
    "data/raw/images",
    "data/split",
    "model/benchmark",
    "model/finetuned",
]

for folder in list_of_files:
    os.makedirs(folder, exist_ok=True)