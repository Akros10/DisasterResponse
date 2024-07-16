# DisasterResponse
udacity project
# ETL pipeline (clean data and store in database)
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
# ML pipeline (loads data, trains classifier and store)
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
