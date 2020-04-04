# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    
    """
    Function that loads the data from the database. In our case
    it is loading our table from SQLite database and transforming the data
    which can be used for training
    
    Input :
    - database_filepath : Location of database file
    
    Output:
     - X : THe dataframe that just has Messages
     - Y : Dataframe that has all the message labels
    
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('final_table', engine)  
    X = df["message"]
    y = df.drop(["id", "message","original", "genre"], axis = 1)
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    """
    Function that takes in plain message and cleans the messages 
    returning the list of cleaned words in the message
    
    Input :
    - text : input text message
    
    output:
    - clean_tokens : tokenized and lemmatized text(a list of words)
  
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = [ w for w in word_tokenize(text) if w not in nltk.corpus.stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    We have a complete Machine learning Pipeline in this function.
    This machine pipeline should take in the message column as input and 
    output classification results on the other 36 categories in the dataset.
    
    Output :
    - pipeine : ML pipeline
    
    """
    pipeline = Pipeline([
        
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    
    # commenting the Gridsearch Part as it will take a lot of time
    #I have used different combinations and passed in the best parameters
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Returns test accuracy, recall, precision and F1 Score.
    
    Inputs:
    model: model object. Instanciated model.
    X_test: pandas dataframe containing test features.
    y_test: pandas dataframe containing test labels.
    category_names: list of strings containing category names.
    
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))
          


def save_model(model, model_filepath):
    
    """
    Function to save the trained model to a pickle file
    
    Input:
    - model : The trained and evaluated ML model
    - model : location to save the pickle file
    
    """
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()