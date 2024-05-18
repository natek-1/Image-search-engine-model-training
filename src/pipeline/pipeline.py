import os, sys

import torch
from tqdm import tqdm
from from_root import from_root
from torch.utils.data import DataLoader

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model import NeuralNet
from src.components.trainer import Trainer
from src.components.embeddings import ImageFolder, EmbeddingGenerator
from src.components.nearest_neighbours import Annoy
from src.exception import CustomException
from src.logger import logging


class Pipeline:

    def __init__(self):
        self.paths = ["data", "data/raw", "data/split", "data/embeddings",
                "model", "model/benchmark", "model/finetuned"]
        self.device = "cpu"
        
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
    
    def initiate_data_ingestion(self):
        try:

            for folder in self.paths:
                path = os.path.join(from_root(), folder)
                if not os.path.exists(path):
                    os.mkdir(path)

            data_ingestion = DataIngestion()
            data_ingestion.run_step()
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def initiate_data_preprocessing(self):
        try:

            data_preprocessing = DataPreprocessing()
            data_preprocessing.run_step()
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def initiate_model(self):
        try:
            return NeuralNet()
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

    def initiate_model_training(self, loaders, net):
        try:
            trainer = Trainer(self.device, loaders, net)
            trainer.train_model()
            trainer.evaluate(False)
            trainer.save_model()
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    
    
