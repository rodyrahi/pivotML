import streamlit as st
import pandas as pd

from genrate_models import *

def main():
    st.set_page_config(page_title="PivotML", page_icon="🤖")

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

        # Create two columns for better layout
        col1, col2 = st.columns(2)
    
        with col1:
            is_regression = st.checkbox("Is this a regression model?", help="Check this if your target variable is continuous")

        # Read the CSV file
        df = pd.read_csv(uploaded_file)
    
        # Get columns from the dataframe
        columns = df.columns.tolist()

        with col2:
            target_column = st.selectbox("Select Target Column", columns, help="Choose the column you want to predict")

        # Select features with better description
        feature_columns = st.multiselect(
            "Select Feature Columns",
            columns,
            help="Choose the columns you want to use as features for prediction"
        )
    
        if st.button("🎯 Generate Models"):
            with st.spinner('Training models... Please wait...'):
                df = filter_dtype(df)
                X, y = genrate_X_y(df, target_column)
                X_train, X_test, y_train, y_test = genrate_train_test_split(X, y, 0.4, 1)

                automl = SimpleAutoML()
                results = automl.train_evaluate_all(X_train, X_test, y_train, y_test)
                best_model_name, best_model, best_score = automl.get_best_model(X_train, X_test, y_train, y_test)

                # Display results in a more organized way
                st.subheader("Model Results 📊")
                for model_name, metrics in results.items():
                    st.metric(label=model_name, value=f"Score: {metrics}")

                st.success(f"Best Model: {best_model_name} with score: {best_score:.4f} 🏆")

if __name__ == "__main__":
    main()
