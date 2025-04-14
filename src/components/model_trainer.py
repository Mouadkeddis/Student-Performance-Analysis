import sys
import os
from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object,evaluate_models
@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split  training and test input data")
            X_train, y_train,X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "adaboost regressor" : AdaBoostRegressor(),
                "Gradient Booster" : GradientBoostingRegressor(),
                "K-Nieghbors classifier" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),

            }
            params={
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128,256],

                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                   # "splitter": ["best", "random"],
                },
                "Gradient Booster": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                },
                "Linear Regression": {},
                "K-Nieghbors classifier": {
                    "n_neighbors": [ 5, 7, 9, 11],
                   # "weights": ["uniform", "distance"],
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    #"subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                },
                "CatBoosting Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "depth": [6, 8, 10],
                    #"subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                    "iterations": [30, 50, 100],
                },
                "adaboost regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    #"subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                },
            }
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models=models,param=params)
            #TO get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            #TO get the best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")

            # save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
