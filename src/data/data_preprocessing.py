import pandas as pd
import numpy as np
import yaml
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')
nltk.download('stopwords')

import logging

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def lemmatization(text: str) -> str:
        lemmatizer= WordNetLemmatizer()

        text = text.split()

        text=[lemmatizer.lemmatize(y) for y in text]

        return " " .join(text)
    
def remove_stop_words(text:str) -> str:
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text: str) -> str:
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text:str) -> str:
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df


def fetch_data (data_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(os.path.join(data_url, 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_url, 'test.csv'))
        logger.debug('Data loaded successfully')
    except :
        logger.error('File not found at {}'.format(data_url))
        raise
    return train_data, test_data

def transform_data (train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    return train_processed_data, test_processed_data

def save_data (data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_processed_data.csv"))
        test_data.to_csv(os.path.join(data_path, "test_processed_data.csv"))
        logger.debug('Data saved successfully')
    except:
        logger.error(f"Error: An unexpected error occurred while saving the data.")
        raise
    
def main():
    try:
        train_data, test_data = fetch_data('./data/raw')
        train_processed_data, test_processed_data = transform_data(train_data, test_data)
        data_path = os.path.join("data","interim")
        save_data(data_path, train_processed_data, test_processed_data)
        logger.debug('data preprocessing step complete successfully')
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process : {}".format(e))
        raise
        
if __name__ == '__main__':
    main()