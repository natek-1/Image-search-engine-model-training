import sys
import os
from typing import Dict

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.entity.config_entity import TrainerConfig
from src.components.model import NeuralNet
from src.components.data_preprocessing import DataPreprocessing
from src.exception import CustomException
from src.logger import logging


class Trainer:

    def __init__(self, device: str, loader: Dict, model: NeuralNet):
        self.config: TrainerConfig = TrainerConfig()
        self.device = device
        self.train_loader: DataLoader = loader["train_data_loader"][0]
        self.valid_loader: DataLoader = loader["valid_data_loader"][0]
        self.test_loader: DataLoader = loader["test_data_loader"][0]
        self.criteion = nn.CrossEntropyLoss()
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr = 1e-4)
        self.evaluation = self.config.evaluation
    

    def train_model(self):
        '''
        Method that would train the model give at initialization
        '''
        try:
            logging.info("Starting the training process")

            # Starting the training
            for epoch in range(self.config.epoch):
                print(f"Epoch Number: {epoch}")
                logging.info(f"Epoch Number: {epoch}")

                running_loss = 0.0
                running_correct = 0
                
                for loader in tqdm(self.train_loader):
                    data, target = loader[0].to(self.device), loader[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = self.criteion(outputs, target)
                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    running_correct += (preds == target).sum().item()

                    loss.backward()
                    self.optimizer.step()
                
                loss = running_loss / len(self.train_loader.dataset)
                accuracy = 100. * (running_correct / len(self.train_loader.dataset))

                val_loss, val_acc = self.evaluate()

                print(f"Train Acc : {accuracy:.2f}%, Train Loss : {loss:.4f} Validation Acc : {val_acc:.2f}%, Validation Loss : {val_loss:.4f}")
                logging.info(f"Train Acc : {accuracy:.2f}%, Train Loss : {loss:.4f} Validation Acc : {val_acc:.2f}%, Validation Loss : {val_loss:.4f}")
            
            logging.info("Done with training process")
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def evaluate(self, validation = True):
        '''
        method that evaluates the model current as using either the validation or test data
        input: evaluation: bool to decide if the test or validation data is used
        output: eval_loss and eval accracy
        '''
        try:
            self.model.eval()
            eval_acc = []
            eval_loss = []

            loader = self.valid_loader if validation else self.test_loader
            print("starting evaluation")

            with torch.inference_mode():
                for batch in tqdm(loader):
                    data, target = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.model(data)
                    loss = self.criteion(logits, target)
                    eval_loss.append(loss.item())
                    preds = torch.argmax(logits, dim=1).flatten()
                    accuracy = (preds == target).cpu().numpy().mean() * 100
                    eval_acc.append(accuracy)
            eval_loss = np.mean(eval_loss)
            eval_acc = np.mean(eval_acc)

            return eval_loss, eval_acc
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
    
    def save_model(self):
        try:
            logging.info(f"model is to be stored at {self.config.PATH}")
            torch.save(self.model.state_dict(), self.config.PATH)
            logging.info("model is saved")
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

if __name__ == "__main__":
    try:
        data_preprocessing = DataPreprocessing()
        loader = data_preprocessing.run_step()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        trainer = Trainer(device, loader, NeuralNet())
        trainer.train_model()
        print(trainer.evaluate(False))
        trainer.save_model()
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error.error_message)
        raise error

