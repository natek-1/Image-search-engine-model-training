import sys, json
from typing_extensions import Literal

from tqdm import tqdm
from annoy import _Vector, AnnoyIndex


from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import AnnoyConfig
from src.exception import CustomException
from src.logger import logging


# the first few lines of each method are copied from the documentation
class CustomAnnoy(AnnoyIndex):

    def __init__(self, f: int,  metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        super().__init__(f, metric)
        self.label = []
    
    def add_item(self, i: int, vector, label: str) -> None:
        super().add_item(i, vector)
        self.label.append(label)
    
    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels = [self.label[link] for link in indexes]
        return labels

    def load(self, fn: str, prefault: bool = ...):
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...):
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Annoy(object):

    def __init__(self):
        self.config: AnnoyConfig = Annoy()
        self.mongo: MongoDBClient= MongoDBClient()
        self.result = self.mongo.get_collection_document()["info"]

    
    def build_annoy_format(self):
        try:
            Ann = CustomAnnoy(256, "euclidean")
            logging.info("Building custo annoy")
            for i, record in tqdm(enumerate(self.result)):
                Ann.add_item(i, record["images"], record["s3_link"])
            logging.info("added items to custom annoy")
            Ann.build(100)
            Ann.save(self.config.EMBEDDING_PATH)
            logging.info(f"annoy was saved to {self.config.EMBEDDING_PATH}")
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error
        
    def run_step(self):
        self.build_annoy_format()


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()



    
