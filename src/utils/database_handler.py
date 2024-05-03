from typing import List, Any, Dict
import sys

from pymongo import MongoClient
from dotenv import load_dotenv


from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataBaseConfig


load_dotenv()

class MongoDBClient:
    def __init__(self):
        self.config: DataBaseConfig = DataBaseConfig()
        self.client: MongoClient = MongoClient(self.config.URL)
    
    def insert_bulk_records(self, documents: List[Dict[str, Any]]):
        '''
        method that will allow to insert the embeddings into the database
        input document: a list containing the embeddings that will be given the databse
        output: a dictionary to let the caller know the status of the insertion
        '''
        try:
            logging.info("about to insert inserting bulk records")
            db = self.client[self.config.DATABASENAME]
            collection = self.config.COLLECTION

            # make sure the collect exists in the databse
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            result = db[collection].insert_many(documents)
            logging.info("done inserting bulk documents")
            return {"Response": "Success", "Inserted Documents": len(result.inserted_ids)}
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    def get_collection_document(self):
        '''
        method to get the embeddings from the databse
        output: a dictionay with the status and the ebeddings
        '''
        try:
            logging.info("about to retrive documents from records")
            db = self.client[self.config.DATABASENAME]
            collection = self.config.COLLECTION
            result = db[collection].find()
            logging.info("done getting document from database")
            return {"Response": "Success", "Info": result}
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    def drop_collection(self):
        '''
        method to drop collection from a out database (mostly used for testing)
        output: dictionary with status to indicate if the drop was successful
        '''
        try:
            logging.info("about drop collection from databse")
            db = self.client[self.config.DATABASENAME]
            collection = self.config.COLLECTION
            db[collection].drop()
            logging.info("done dropping collection")
            return {"Response": "Success"}
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.str)
            raise error

if __name__ == "__main__":
    data = [
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"}
    ]

    mongo = MongoDBClient()
    print(mongo.insert_bulk_records(data))
    result = mongo.get_collection_document()
    for i in result["Info"]:
        print(i)
    print(mongo.drop_collection())

