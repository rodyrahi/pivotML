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
    
    return 'Target column updated successfully', 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)