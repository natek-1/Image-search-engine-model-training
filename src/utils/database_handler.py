from typing import List, Tuple, Any

from pymongo import MongoClient
from dotenv import load_dotenv


from src.exeception import CustomException
from src.logger import logging
from src.entity import DataBaseConfig


load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.config: DataBaseConfig = DataBaseConfig()
        self.client: MongoClient = MongoClient(self.config.URL)
    
    #def insert_bulk_records(self, documents: List[])
