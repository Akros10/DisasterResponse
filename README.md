# Project: Disaster Response Pipeline
In this project, you'll apply these skills to analyze disaster data from Appen(opens in a new tab) (formerly Figure 8) to build a model for an API that classifies disaster messages.

There are 3 main parts:
1. ETL Pipeline to import messages and categories and store them in a cleaned database.
2. ML Pipeline, for training an ML to classify messages.
3. WebApp to show the classification for every message as an input.
4. 
# How to use it?
## ETL pipeline (clean data and store in database)
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
## ML pipeline (loads data, trains classifier and store)
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
## To run the app
Run the following command
python app/run.py
Go to http://0.0.0.0:3001/
