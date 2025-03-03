from flask import Flask, render_template
import json
import pandas as pd

df = pd.read_csv('creditscore.csv')




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
        
        targets = df.columns.tolist()
        return render_template('targets.html', targets=targets)






@app.route('/column/<column_name>')
def column(column_name):
    if column_name not in df.columns:
        return f"Column '{column_name}' not found.", 404

    distribution = df[column_name].value_counts().to_dict()
    print(distribution)
    return render_template('column.html', column_name=column_name, distribution=distribution)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)