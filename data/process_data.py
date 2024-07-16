# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:34:04 2024

@author: GarciaDan
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(msg_path, cat_path):
    """
    Load csv and merge them to one unique df
    In:
    msg_path - path messages csv file
    cat_path - path categories csv file
    
    Out:
    df - Merged data
    """
    messages = pd.read_csv(msg_path)
    categories = pd.read_csv(cat_path)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    """
    Get the unify data and clean it for proper use
    In: df - from load data
    Out: df - cleaned data
    """
    #Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from 'df'
    df = df.drop('categories', axis = 1)
    
    # concatenate the original datafram with the new 'categories' dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
       
    # I found that related column has a max value of 2, remove it
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
        
    return df


def save_data(df, db_filename):
    """
    save df into a sqllite database
    In: df - cleaned data
    db_filename - db filename for sqlite 
    
    """
    engine = create_engine('sqlite:///'+ db_filename)
    df.to_sql('DisasterResponse_table', engine, index = False, if_exists = 'replace')


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
