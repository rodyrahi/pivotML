import streamlit as st
import pandas as pd

from genrate_models import *

def main():
    st.set_page_config(page_title="PivotML", page_icon="🤖" , layout="wide")

    # st.set_page_config()


    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("PivotML 🤖")

    st.header("Let's Make Some ML Models 🚀")

    # Add description
    st.markdown("Upload your CSV file and let's create some machine learning models!")

    # File uploader with better styling
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Please upload a CSV file containing your dataset")

    if uploaded_file is not None:
        st.success("File uploaded successfully! 👍")

        tab_basic, tab_advanced = st.tabs(["Basic Settings", "Advanced Settings"])
        
        with tab_basic:

            col1, col2 = st.columns(2)
        
            with col1:
                is_regression = st.checkbox("Is this a regression model?", help="Check this if your target variable is continuous")
                test_split = st.slider(
                        "Select Test-Train Split Range",
                        min_value=1,
                        max_value=99,
                        value=(40),  # Default range values
                        step=1,  # Allows selection in 1% increments
                        format="%.0f%%"  # Displays values as percentages
                    )
                test_split = test_split/100
                print(test_split)

            # Read the CSV file
            df = pd.read_csv(uploaded_file)
        
            # Get columns from the dataframe
            columns = df.columns.tolist()

            with col2:
                target_column = st.selectbox("Select Target Column", columns, help="Choose the column you want to predict")
            print(target_column)
            target_column = target_column
            print(columns)
            columns.remove(target_column)
            
            
            print(columns)
            feature_columns = st.multiselect(
                "Select Feature Columns",
                columns,
                help="Choose the columns you want to use as features for prediction"
            )


        
        with tab_advanced:

            with st.expander("⚙️ General Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    random_state = st.number_input("Random Seed 🎲", 
                        min_value=1, 
                        max_value=1000, 
                        value=42,
                        help="Set seed for reproducible results")
                    
                    cross_val_folds = st.number_input("Cross Validation Folds 📊", 
                        min_value=2, 
                        max_value=10, 
                        value=5,
                        help="Number of folds for cross-validation")
                
                with col2:
                    n_jobs = st.number_input("Number of CPU Cores 💻", 
                        min_value=1, 
                        max_value=8, 
                        value=4,
                        help="Number of CPU cores to use for parallel processing")
                    
                    verbose = st.checkbox("Enable Verbose Output 📝", 
                        value=True,
                        help="Show detailed progress during model training")

            with st.expander("🔍 Feature Engineering"):
                handle_missing = st.selectbox("Handle Missing Values",
                    ["mean", "median", "most_frequent", "constant"],
                    help="Method to handle missing values in the dataset")
                
                feature_scaling = st.selectbox("Feature Scaling Method",
                    ["standard", "minmax", "robust", "none"],
                    help="Method to scale the features")
                
                categorical_encoding = st.selectbox("Categorical Encoding",
                    ["label", "onehot", "target", "none"],
                    help="Method to encode categorical variables")
                
            with st.expander("🧠 Neural Network Settings"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            n_layers = st.number_input("Number of Hidden Layers 🔢", 
                                min_value=1, 
                                max_value=5, 
                                value=2,
                                help="Number of hidden layers in the neural network")
                            
                            neurons_per_layer = st.number_input("Neurons per Hidden Layer 🔄", 
                                min_value=4, 
                                max_value=256, 
                                value=64,
                                help="Number of neurons in each hidden layer")
                            
                            activation = st.selectbox("Activation Function ⚡",
                                ["relu", "tanh", "sigmoid"],
                                help="Activation function for hidden layers")
                        
                        with col2:
                            epochs = st.number_input("Number of Epochs 🔄", 
                                min_value=10, 
                                max_value=1000, 
                                value=100,
                                help="Number of complete passes through the training dataset")
                            
                            batch_size = st.number_input("Batch Size 📦", 
                                min_value=8, 
                                max_value=256, 
                                value=32,
                                help="Number of samples processed before model update")
                            
                            learning_rate = st.number_input("Learning Rate 📈", 
                                min_value=0.0001, 
                                max_value=0.1, 
                                value=0.001, 
                                format="%.4f",
                                help="Step size for model parameter updates")       

            with st.expander("🌲 Random Forest Settings"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                n_estimators = st.number_input("Number of Trees 🌳", 
                                    min_value=10, 
                                    max_value=1000, 
                                    value=100,
                                    help="Number of trees in the forest")
                                
                                max_depth = st.number_input("Maximum Depth 📏", 
                                    min_value=1, 
                                    max_value=50, 
                                    value=10,
                                    help="Maximum depth of each tree")
                            
                            with col2:
                                min_samples_split = st.number_input("Minimum Samples Split 🔀", 
                                    min_value=2, 
                                    max_value=20, 
                                    value=2,
                                    help="Minimum samples required to split internal node")
                                
                                min_samples_leaf = st.number_input("Minimum Samples Leaf 🍃", 
                                    min_value=1, 
                                    max_value=20, 
                                    value=1,
                                    help="Minimum samples required at leaf node")
            
            with st.expander("🎯 XGBoost Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    learning_rate_xgb = st.number_input("Learning Rate (XGB) 📈", 
                        min_value=0.01, 
                        max_value=1.0, 
                        value=0.1, 
                        format="%.2f",
                        help="Learning rate for XGBoost")
                    
                    max_depth_xgb = st.number_input("Maximum Depth (XGB) 📏", 
                        min_value=1, 
                        max_value=20, 
                        value=6,
                        help="Maximum depth of XGBoost trees")
                
                with col2:
                    n_estimators_xgb = st.number_input("Number of Estimators (XGB) 🔢", 
                        min_value=10, 
                        max_value=1000, 
                        value=100,
                        help="Number of gradient boosted trees")
                    
                    subsample = st.number_input("Subsample Ratio 📊", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=1.0, 
                        format="%.1f",
                        help="Subsample ratio of training instances")

            with st.expander("🎲 SVM Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    kernel = st.selectbox("Kernel Type 🔄",
                        ["rbf", "linear", "poly", "sigmoid"],
                        help="Type of kernel function")
                    
                    C = st.number_input("Regularization Parameter (C) 📊", 
                        min_value=0.1, 
                        max_value=10.0, 
                        value=1.0, 
                        format="%.1f",
                        help="Regularization parameter")
                
                with col2:
                    gamma = st.selectbox("Kernel Coefficient (Gamma) 📈",
                        ["scale", "auto"],
                        help="Kernel coefficient for RBF, Poly and Sigmoid kernels")
                    
                    degree = st.number_input("Polynomial Degree 📐", 
                        min_value=1, 
                        max_value=5, 
                        value=3,
                        help="Degree of polynomial kernel function")


    col1, col2, col3 = st.columns([1, 1, 1])  # Create columns to center the button
    with col1:
  
        gen_button = st.button("🎯 Generate Models")

    if gen_button:
        with st.spinner('Training models... Please wait...'):
            df = filter_dtype(df)
            X, y = genrate_X_y(df, target_column)
            # X = min_max_scale(X)
            X_train, X_test, y_train, y_test = genrate_train_test_split(X, y, test_split, 1)
            X_train , X_test = min_max_scale(X_train , X_test)



            automl = SimpleAutoML()
            results = automl.train_evaluate_all(X_train, X_test, y_train, y_test)
            best_model_name, best_model, best_score = automl.get_best_model(X_train, X_test, y_train, y_test)

            # Display model results
            st.subheader("Model Results 📊")

            # Create a container to hold the results
            with st.container():
                cols = st.columns(len(results))  # Create columns for each model
                
                for col, (model_name, score_data) in zip(cols, results.items()):
                    score = score_data['Accuracy']  # Extract score
                    
                    print(score)

                    with col:
                        html_card = f'''<div style="
                                        padding: 10px; margin:10px; border: 2px solid #FF4B4B; 
                                        background-color:#262730;
                                        border-radius: 10px; text-align: center; 
                                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);"> 
                                        <h4 style="">{model_name}</h4> 
                                        <p style="font-size: 20px; font-weight: bold; color: #0098C3;">Accuracy: {score:.4f}</p>
                                        </div>'''
                        print(html_card) 
                        st.markdown(html_card, unsafe_allow_html=True)

            st.success(f"🏆 Best Model: {best_model_name} with score: {best_score:.4f}")



if __name__ == "__main__":
    main()
