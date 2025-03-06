import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



from sklearn.neural_network import MLPClassifier , MLPRegressor
from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier ,  DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier ,   RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score , f1_score


def filter_dtype(df, threshold=0.95):
    df = df.copy()  
    
    for col in df.columns:
        if df[col].dtype == object: 
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            numeric_count = numeric_col.notna().sum()
            string_count = len(df[col]) - numeric_count

            if numeric_count > 0:
                if string_count > len(df[col]) * threshold:
                    df[col] = df[col].astype(str)  
                else:
                    df[col] = numeric_col  

    
    df.dropna(inplace=True)  
    
    return df

def genrate_X_y(df, target_column, features=[] ):

    X = df.drop([target_column], axis=1)
    
    if len(features) > 0:
        
        X = X[features]
        print(X)
    for col in X.columns:
        
        if X[col].dtype == 'object': 
            if X[col].nunique() < 10:
                X[col] = X[col].astype('category')
            else:
                X.drop(col, axis=1, inplace=True)
      
    y = df[target_column]

    return X, y

def min_max_scale(X_train , X_test):
    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train , X_test

    

def genrate_train_test_split(X, y , test_size=0.4 , random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    return X_train, X_test, y_train, y_test





class SimpleAutoML:
    def __init__(self ,hidden_layer_sizes = (5, 3) , is_regression=False):
       
        self.is_regression = is_regression
        self.hidden_layer_sizes = hidden_layer_sizes

        
        self.modelsClassifiers = {
            'Neural Network': MLPClassifier( hidden_layer_sizes= self.hidden_layer_sizes),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            
        }
        self.modelsReggression = {
            'Neural Network': MLPRegressor(hidden_layer_sizes= self.hidden_layer_sizes),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            # 'Logistic Regression': LogisticRegression(),
            'Linear Regression': LinearRegression()
        }
        
    def train_evaluate_all(self, X_train, X_test, y_train, y_test , is_regression = False):
        

        models = {}
        is_regression = self.is_regression
        results = {}
        
     
        # self.models['Neural Network'] = MLPClassifier()
        
        
        if is_regression:
            for name, model in self.modelsReggression.items():
                # Train the model
                model.fit(X_train, y_train)
                
                
                y_pred = model.predict(X_test)
                
            
                score = r2_score(y_test, y_pred)
                metric_name = 'R2 Score'
                results[name] = {metric_name: score}
        else:

            for name, model in self.modelsClassifiers.items():
             
                # Train the model
                model.fit(X_train, y_train )
                
                models.update({name: model})
                y_pred = model.predict(X_test)
                

                test_accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = accuracy_score(y_train, model.predict(X_train))
                metric_name = 'Accuracy'

                f1score = f1_score(y_test, y_pred , average='weighted')



                results[name] = {"Test Accuracy": test_accuracy , "Train Accuracy" : train_accuracy  , 'F1 Score': f1score }
            
        return results , models

    
     
    def get_best_model(self, X_train, X_test, y_train, y_test):
        results , models = self.train_evaluate_all(X_train, X_test, y_train, y_test)
        best_score = -float('inf')
        best_model = None
        
        for name, metrics in results.items():
            score = list(metrics.values())[0]
            if score > best_score:
                best_score = score
                best_model = name
                
        return best_model, self.modelsClassifiers[best_model], best_score






# automl = SimpleAutoML()
# results = automl.train_evaluate_all(X_train, X_test, y_train, y_test)
# best_model_name, best_model, best_score = automl.get_best_model(X_train, X_test, y_train, y_test)

# print("Results for all models:")
# for model_name, metrics in results.items():
#     print(f"{model_name}: {metrics}")

# print(f"\nBest model: {best_model_name}")
# print(f"Best score: {best_score:.4f}")
