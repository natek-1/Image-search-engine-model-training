import os, sys

import torch
from tqdm import tqdm
from from_root import from_root
from torch.utils.data import DataLoader

from src.utils.storage_handler import S3Connector
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
            loaders = data_preprocessing.run_step()
            return loaders
        
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
    
    def generate_embeddings(self, loader: dict, model: torch.nn.Module):
        try:
            for name, val in loader.items():         
                data = ImageFolder(label_map=val[1].class_to_idx)
                dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
                logging.info(f"currently generating embeddings for {name} dataset")
                embeds = EmbeddingGenerator(model=model, device=self.device)
                for batch_num, values in tqdm(enumerate(dataloader)):
                    img, target, link = values
                    embeds.run_step(batch_num, img, target, link)
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def create_annoy(self):
        try:
            ann = Annoy()
            ann.run_step()
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def push_artifacts(self):
        try:
            connection: S3Connector = S3Connector()
            connection.zip_file()
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def run_pipeline(self):
        try:
            logging.info("staring the entire pipeline")


            self.initiate_data_ingestion()

            logging.info("data ingestion done")

            loaders = self.initiate_data_preprocessing()

            logging.info("done getting the data loaders")

            model = self.initiate_model()

            logging.info("model was generated")
            self.initiate_model_training(loaders, model)

            logging.info("the model was generated")

            self.generate_embeddings(loaders, model)

            logging.info("the embeddings for images was generated")
            self.create_annoy()

            logging.info("Our custom anony was trained")
            self.push_artifacts()

            logging.info("the files were uploaded to aws")

            logging.info("done running the entire pipeline")
            
            
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    

if __name__ == "__main__":
    image_search = Pipeline()
    image_search.run_pipeline()
