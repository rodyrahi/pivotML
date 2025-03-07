from flask import Blueprint, render_template
import pandas as pd

from data.database import *

from data.sql_database import *


source_file = 'Iris.csv'
# global df
df = pd.read_csv(source_file)

features_use = [(feature, True) for feature in df.columns.tolist()]
algorithms_use = [('Random Forest' , True), ('Decision Tree' , True),
                  ('Logistic Regression' , True) ,('Linear Regression' , True), ('SVM' , True) , ('KNN' , True), 
                  ('Neural Network' , True)]

target_column = ''



testtrainsplit = 0.4

# user_id, csv_path, dataframe, features_used, target_column, testtrainsplit, algorithms_used, metadata


# save_project('raj' , 'iris.csv' , '' , features_use , target_column , testtrainsplit , algorithms_use , {})





main_routes = Blueprint("main_routes", __name__)






@main_routes.route('/')
def project():
    project = {
        'name': 'Credit Scoring',
        'description': 'Predict the credit score of a person based on various features.',
    }
    return render_template('project.html', project=project)

@main_routes.route('/index')
def home():
    return render_template('index.html')
