import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
import pandas as pd
from src.components.model_trainer import ModelTrainer,ModeltrainerConfig
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_set_path : str=os.path.join('artifacts','train.csv')
    test_set_path : str=os.path.join('artifacts','test.csv')
    raw_set_path : str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        #this function is to read the data from it source
        logging.info("Entered the data ingestion method or component")
        try:
            df =pd.read_csv('notebook\data\stud.csv')
            logging.info("Reading the data ad dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_set_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_set_path,index=False,header=True)
            logging.info("train test split initiation")

            train_set, test_set= train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_set_path,index = False, header=True)
            test_set.to_csv(self.ingestion_config.test_set_path,index = False, header=True)

            logging.info('ingestion of data is completed')
            return(
                self.ingestion_config.train_set_path,
                self.ingestion_config.test_set_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ =="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))