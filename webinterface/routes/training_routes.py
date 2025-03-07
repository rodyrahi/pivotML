from flask import Blueprint, render_template
from data.sql_database import *
from genrate_models import *
from genrate_models import genrate_X_y, genrate_train_test_split, min_max_scale, SimpleAutoML






training_routes = Blueprint("training_routes", __name__)

@training_routes.route('/train')
def train():
    target_column = get_project_value('1', 'target_column')
    features_use = get_project_value('1', 'features_used')
    source_file = get_project_value('1', 'csv_path')
    testtrainsplit = get_project_value('1', 'testtrainsplit')
    df = pd.read_csv(source_file)
    df = filter_dtype(df)
    print(features_use)
    features_use_novalues = [feature for feature, isused in features_use if isused ]
    if target_column in features_use_novalues:
            features_use_novalues.remove(target_column)
    
    print(features_use_novalues)
    X, y = genrate_X_y(df, target_column ,  features_use_novalues)
    # X = min_max_scale(X)
    X_train, X_test, y_train, y_test = genrate_train_test_split(X, y, testtrainsplit, 1)
    X_train , X_test = min_max_scale(X_train , X_test)



    automl = SimpleAutoML()
    results , models = automl.train_evaluate_all(X_train, X_test, y_train, y_test)
    best_model_name, best_model, best_score = automl.get_best_model(X_train, X_test, y_train, y_test)

    
    return render_template('results.html' , results = results)

