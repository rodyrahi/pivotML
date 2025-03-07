from flask import Blueprint, render_template
from data.sql_database import *
import pandas as pd

feature_routes = Blueprint("feature_routes", __name__)

df = pd.read_csv(get_project_value('1', 'csv_path'))
features_use = get_project_value('1', 'features_used')

@feature_routes.route('/feature_handling')
def feature_handling():
   
    return render_template('feature_handling.html', targets=features_use)






@feature_routes.route('/column/<column_name>')
def column(column_name):
    
    if column_name not in df.columns:
        return f"Column '{column_name}' not found.", 404

    distribution = df[column_name].value_counts().to_dict()
    boxplot_data = df[column_name].tolist()
    
    return render_template('column.html', column_name=column_name, distribution=distribution, boxplot_data=boxplot_data)





@feature_routes.route('/add/<column_name>')
def add(column_name):
    for i, (feature, _) in enumerate(features_use):
        if feature == column_name:
            features_use[i] = (feature, True)

    update_project_value('1', 'features_used', features_use)
    return '', 200

@feature_routes.route('/remove/<column_name>')
def remove(column_name):
    for i, (feature, _) in enumerate(features_use):
        if feature == column_name:
            features_use[i] = (feature, False)

    update_project_value('1', 'features_used', features_use)
    return '', 200
