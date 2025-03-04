from flask import Flask, render_template
import json
import pandas as pd

df = pd.read_csv('creditscore.csv')

features_use = [(feature, True) for feature in df.columns.tolist()]
algorithms_use = [('Random Forest' , True), ('Decision Tree' , True),
                  ('Logistic Regression' , True) ,('Linear Regression' , True), ('SVM' , True) , ('KNN' , True), 
                  ('Neural Network' , True)]


app = Flask(__name__)

@app.route('/')
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
        # print(features_use)
        targets = df.columns.tolist()
        return render_template('targets.html', targets=features_use)


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

@app.route('/column/<column_name>')
def column(column_name):

    
    if column_name not in df.columns:
        return f"Column '{column_name}' not found.", 404

    distribution = df[column_name].value_counts().to_dict()
    print(distribution)
    return render_template('column.html', column_name=column_name, distribution=distribution)



@app.route('/algorithms')
def algorithms():

    algorithms_labels = ['Random Forest', 'Decision Tree', 'Logistic Regression' ,'Linear Regression', 'SVM' , 'KNN', 'Neural Network']
    return render_template('algorithms.html', algorithms=algorithms_labels)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)