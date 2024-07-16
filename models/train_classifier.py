"""
Created on Mon Jul 15 16:53:25 2024

@author: GarciaDan
"""

import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(db_filepath):
    """
    Import table and separate the message from the category
    In: db_filepath 
    
    Out:
    X - messages 
    y - categories 
    category_names
    """
    engine = create_engine('sqlite:///' + db_filepath)
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    categories =  y.columns.tolist()
    return X, y, categories

def tokenize(text):
    """
    Do the tokenization of the message recevived
    In: text - original message
    
    Out:  clean_tokens - tokenized messages
    """
    #norm
    text= text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model(clf = AdaBoostClassifier()):
    """
    In:  clf - classifier model, if none, AdaBoost
    Out:  cv = model prepared for training
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(clf))])
        
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [0.01, 0.1, 1],
        'clf__estimator__algorithm': ['SAMME', 'SAMME.R']
    }
            
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)
    
    return cv
    
def evaluate_model(model, X_test, Y_test, categories):
    """
    Evaluate model with test samples
    In:
    model - ML model
    X_test - test messages
    y_test - categories 
    categories - categories names
    
    Out:
    print scores (precision, recall, f1-score) 
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=categories))
    

def save_model(model, mdl_filepath):
    """
    Save mdl as pkl
    In:
    model - ML model
    mdl_filepath - location
    
    """
    with open(mdl_filepath, 'wb') as f:
        pickle.dump(model, f)


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
