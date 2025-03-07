
import pandas as pd


source_file = 'Iris.csv'
global df
df = pd.read_csv(source_file)

features_use = [(feature, True) for feature in df.columns.tolist()]
algorithms_use = [('Random Forest' , True), ('Decision Tree' , True),
                  ('Logistic Regression' , True) ,('Linear Regression' , True), ('SVM' , True) , ('KNN' , True), 
                  ('Neural Network' , True)]

target_column = ''



testtrainsplit = 0.4
