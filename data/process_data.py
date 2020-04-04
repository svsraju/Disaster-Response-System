import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import numpy as np
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load our data from the given filepaths. Both the files are read
    in as a pandas Dataframe. The output will be merged dataframe.
    
    Inputs:
    - messages_filepath: location of the disaster_messages.csv file (string)
    - categories_filepath : ocation of the disaster_categories.csv file (string)
    
    Output:
    
   - df - merged dataframe of both the dataframes
    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="inner", on=["id"])
    return df


def clean_data(df):
    """
    Function that cleans the data present in the datagram
    Input:
    - df : The raw uncleaned dataframe
    
    output:
    - df : The dataset which is cleaned and modified as per the requirement
    """
    # creating a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    #renaming the categories columns
    row = categories.iloc[0,:].copy()
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    #converting the values of string to just 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1] 
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #replacing categories colum with new category columns
    df = df.drop("categories", axis = 1)
    df = pd.concat([df, categories], axis=1, sort=False)
    
    #removing dulicates
    
    df = df.drop_duplicates()
    #Removing the rows which have values of 2 in the related column
    df.drop(df[df['related'] == 2].index, inplace = True)
    
    return df


def save_data(df, database_filename):
    """
    Function saves the cleaned dataframe into a sqlite database
    
    Input:
    
    - df : Cleaned dataframe
    - database_filename : The location of the database where the file will be stored
    
    Output :
     None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    
    df.to_sql('final_table', engine, index=False)
    
    
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()