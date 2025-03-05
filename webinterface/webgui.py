from flask import Flask, render_template
import json
import pandas as pd

df = pd.read_csv('creditscore.csv')

features_use = [(feature, True) for feature in df.columns.tolist()]
algorithms_use = [('Random Forest' , True), ('Decision Tree' , True),
                  ('Logistic Regression' , True) ,('Linear Regression' , True), ('SVM' , True) , ('KNN' , True), 
                  ('Neural Network' , True)]

target_column = ''



testtrainsplit = 0.4



app = Flask(__name__)

@app.route('/index')
def home():
    items = [
        {'name': 'Item 1', 'link': '/item1' , 'text': 'Item 1'},


        {'name': 'Item 2', 'link': '/item2' , 'text': 'Item 2'},
        {'name': 'Item 3', 'link': '/item3' , 'text': 'Item 3'},
        {'name': 'Item 4', 'link': '/item4' , 'text': 'Item 4'},
    ]
    return render_template('index.html', items=items)


@app.route('/targets')
def targets():
        
        if target_column == '':
            selected_target = 'No target selected'
        else:
            selected_target = target_column
            
        targets  = df.columns.tolist()
        return render_template('targets.html' , targets = targets , selected_target = selected_target)






@app.route('/add/<column_name>')
def add(column_name):
    for i, (feature, _) in enumerate(features_use):
        if feature == column_name:
            features_use[i] = (feature, True)
    print(column_name , 'added')
    return '',200

@app.route('/remove/<column_name>')
def remove(column_name):
    for i, (feature, _) in enumerate(features_use):
        if feature == column_name:
            features_use[i] = (feature, False)
    print(column_name, 'removed')
    return '', 200

@app.route('/add_algorithm/<algorithm_name>')
def add_algorithm(algorithm_name):
    for i, (algorithm, _) in enumerate(algorithms_use):
        if algorithm == algorithm_name:
            algorithms_use[i] = (algorithm, True)
    print(algorithm_name, 'added')
    return '', 200

@app.route('/remove_algorithm/<algorithm_name>')
def remove_algorithm(algorithm_name):
    for i, (algorithm, _) in enumerate(algorithms_use):
        if algorithm == algorithm_name:
            algorithms_use[i] = (algorithm, False)
    print(algorithm_name, 'removed')
    return '', 200




@app.route('/column/<column_name>')
def column(column_name):
    
    if column_name not in df.columns:
        return f"Column '{column_name}' not found.", 404

    distribution = df[column_name].value_counts().to_dict()
    boxplot_data = df[column_name].tolist()
    
    return render_template('column.html', column_name=column_name, distribution=distribution, boxplot_data=boxplot_data)



@app.route('/algorithms')
def algorithms():

    return render_template('algorithms.html', algorithms=algorithms_use)








@app.route('/testtrain')
def testtrain():
    print(testtrainsplit)
    return render_template('testtrain.html' , testtrainsplit = testtrainsplit)

@app.route('/update_split/<float:split_value>')
def update_split(split_value):
    global testtrainsplit
    testtrainsplit = split_value
    print(f'Test-train split updated to: {testtrainsplit}')
    return '', 200

@app.route('/feature_handling')
def feature_handling():
    return render_template('feature_handling.html', targets=features_use)





@app.route('/')
def project():
    project = {
        'name': 'Credit Scoring',
        'description': 'Predict the credit score of a person based on various features.',
        'features': features_use,
        'algorithms': algorithms_use,
       
    }
    return render_template('project.html' , project = project)




@app.route('/add_target/<target_column_selected>')
def add_target(target_column_selected):
    global target_column
    target_column = target_column_selected
    
    return '', 200








@app.route('/algorithm/<selected_algorithm>')
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




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)