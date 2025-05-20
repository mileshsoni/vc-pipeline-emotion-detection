import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging 
from sklearn.base import ClassifierMixin

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(path:str) -> ClassifierMixin:
    try:
        clf = pickle.load(open(path, 'rb'))
        logger.debug('model loaded successfully')
        return clf
    except FileNotFoundError as e:
        logger.error('file not found at {}'.format(path))
        raise
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
    
def load_test_data (path: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(os.path.join(path,'test_bow.csv'))
        logger.debug('test data loaded successfully')
        return test_data
    except FileNotFoundError as e:
        logger.error('File not found at {}'.format(path))
        raise
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
        
def x_y_split(test_data : pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return x_test, y_test

def make_predictions (clf : ClassifierMixin, x_test : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
# Make predictions
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1] # for auc score
        logger.debug('predictions done')
        return y_pred, y_pred_proba
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
    
def calculate_evaluation_metrics(y_test : np.ndarray, y_pred: np.ndarray, y_pred_proba : np.ndarray) -> json:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # output a json file
        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug('Evaluation metrics calculated successfully')
        return metrics_dict
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
    
def save_evaluation_metrics(metrics_dict : json, path : str) -> None:
    try:
        json.dump(metrics_dict, open(path, 'w'))
        logger.debug('evaluation metrics saved successfully')
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
        
def main() -> None:
    try:
        clf = load_model('models/model.pkl')
        test_data = load_test_data('./data/processed')
        x_test, y_test = x_y_split(test_data)
        y_pred, y_pred_proba = make_predictions(clf, x_test)
        metrics_dict = calculate_evaluation_metrics(y_test, y_pred, y_pred_proba)
        save_evaluation_metrics(metrics_dict, 'reports/metrics.json')
        logger.debug('model evaluation step complete successfully')
    except Exception as e:
        logger.error(f"Failed to complete the model evaluation process: {e}")
        raise
if __name__ == "__main__":
    main()
        
