from flask import Blueprint, render_template
from data.sql_database import *
algorithm_routes = Blueprint("algorithm_routes", __name__)

algorithms_use = get_project_value('1', 'algorithms_used')

@algorithm_routes.route('/algorithms')
def algorithms():
    return render_template('algorithms.html', algorithms=algorithms_use)




@algorithm_routes.route('/algorithm/<selected_algorithm>')
def algorithm(selected_algorithm):


    

    random_forest_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split':2,
    'min_samples_leaf': 1,
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'random_state': 42
    }

    decision_tree_params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': ['gini', 'entropy'],
    'random_state': 42
    }

    logistic_regression_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': 100,
    'random_state': 42
    }

    linear_regression_params = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False]
    }

    svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': 2,
    'random_state': 42
    }

    knn_params = {
    'n_neighbors': 3,
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': 2
    }

    neural_network_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': 0.0001,
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': 200,
    'random_state': 42
    }    


    if selected_algorithm == 'Random Forest':
        algorithm_params = random_forest_params
    elif selected_algorithm == 'Decision Tree':
        algorithm_params = decision_tree_params
    elif selected_algorithm == 'Logistic Regression':
        algorithm_params = logistic_regression_params
    elif selected_algorithm == 'Linear Regression':
        algorithm_params = linear_regression_params
    elif selected_algorithm == 'SVM':
        algorithm_params = svm_params
    elif selected_algorithm == 'KNN':
        algorithm_params = knn_params
    elif selected_algorithm == 'Neural Network':
        algorithm_params = neural_network_params

    print(selected_algorithm)
    return render_template('algorithm_settings.html', selected_algorithm=selected_algorithm, algorithm_params=algorithm_params)



@algorithm_routes.route('/add_algorithm/<algorithm_name>')
def add_algorithm(algorithm_name):
    for i, (algorithm, _) in enumerate(algorithms_use):
        if algorithm == algorithm_name:
            algorithms_use[i] = (algorithm, True)

    update_project_value('1', 'algorithms_used', algorithms_use)
    return '', 200

@algorithm_routes.route('/remove_algorithm/<algorithm_name>')
def remove_algorithm(algorithm_name):
    for i, (algorithm, _) in enumerate(algorithms_use):
        if algorithm == algorithm_name:
            algorithms_use[i] = (algorithm, False)

    update_project_value('1', 'algorithms_used', algorithms_use)
    return '', 200
