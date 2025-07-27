import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="PCA vs ICA Comparison on Random Forest",
    page_icon="📊",
    layout="wide"
)

# Title of the app
st.title("Performance Evaluation: PCA vs ICA on Random Forest Classifier")
st.markdown("""
This application compares the performance of Principal Component Analysis (PCA) and 
Independent Component Analysis (ICA) as feature selection techniques when used with 
a Random Forest classifier on diabetes datasets.
""")

# Sidebar configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Default parameters for analysis
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.slider("Random State", 0, 100, 42)

# Specify RF parameters
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
max_depth = st.sidebar.slider("Max Depth", 1, 30, 10, 1)

# Components selection
pca_components = st.sidebar.slider("PCA Components", 2, 10, 5, 1)
ica_components = st.sidebar.slider("ICA Components", 2, 10, 5, 1)

# Function to load default dataset if no file is uploaded
@st.cache_data
def load_default_dataset():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)  # Binary classification
    return X, y, "Default Diabetes Dataset (Sklearn)"

# Function to preprocess the data
def preprocess_data(df, target_column):
    # Extract features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return None, None
    
    # Convert target to categorical if it's not already
    if not pd.api.types.is_categorical_dtype(y) and not pd.api.types.is_bool_dtype(y):
        try:
            y = y.astype(int)
        except:
            st.warning("Converting target to binary (values > mean = 1, else 0)")
            y = (y > y.mean()).astype(int)
    
    return X, y

# Function to run the analysis
def run_analysis(X, y, dataset_name):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Baseline Random Forest (No dimensionality reduction)
    baseline_model, baseline_metrics, baseline_time = train_evaluate_rf(
        X_train_scaled, X_test_scaled, y_train, y_test, "Baseline (No Reduction)"
    )
    
    # PCA dimensionality reduction
    pca_start_time = time.time()
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    pca_transform_time = time.time() - pca_start_time
    
    # Train and evaluate RF with PCA
    pca_model, pca_metrics, pca_train_time = train_evaluate_rf(
        X_train_pca, X_test_pca, y_train, y_test, "PCA"
    )
    pca_total_time = pca_transform_time + pca_train_time
    
    # ICA dimensionality reduction
    ica_start_time = time.time()
    ica = FastICA(n_components=ica_components, random_state=random_state)
    X_train_ica = ica.fit_transform(X_train_scaled)
    X_test_ica = ica.transform(X_test_scaled)
    ica_transform_time = time.time() - ica_start_time
    
    # Train and evaluate RF with ICA
    ica_model, ica_metrics, ica_train_time = train_evaluate_rf(
        X_train_ica, X_test_ica, y_train, y_test, "ICA"
    )
    ica_total_time = ica_transform_time + ica_train_time
    
    # Store metrics and times in a dictionary
    results = {
        "models": {
            "Baseline": baseline_model,
            "PCA": pca_model,
            "ICA": ica_model
        },
        "metrics": {
            "Baseline": baseline_metrics,
            "PCA": pca_metrics,
            "ICA": ica_metrics
        },
        "times": {
            "Baseline": {"total": baseline_time, "transform": 0, "train": baseline_time},
            "PCA": {"total": pca_total_time, "transform": pca_transform_time, "train": pca_train_time},
            "ICA": {"total": ica_total_time, "transform": ica_transform_time, "train": ica_train_time}
        },
        "data": {
            "X_train": X_train_scaled, 
            "X_test": X_test_scaled,
            "X_train_pca": X_train_pca, 
            "X_test_pca": X_test_pca,
            "X_train_ica": X_train_ica, 
            "X_test_ica": X_test_ica,
            "y_train": y_train, 
            "y_test": y_test
        },
        "components": {
            "PCA": pca,
            "ICA": ica
        }
    }
    
    return results

# Train and evaluate Random Forest
def train_evaluate_rf(X_train, X_test, y_train, y_test, method_name):
    start_time = time.time()
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }
    
    train_time = time.time() - start_time
    
    return model, metrics, train_time

# Function to create visualizations for the results
def create_visualizations(results, dataset_name):
    # Create tabs for organizing visualizations
    tabs = st.tabs(["Performance Metrics", "Confusion Matrices", "ROC Curves", "Component Analysis", "Time Comparison"])
    
    with tabs[0]:  # Performance Metrics
        st.header("Performance Metrics Comparison")
        
        # Extract metrics for each method
        methods = ["Baseline", "PCA", "ICA"]
        metrics_df = pd.DataFrame({
            "Method": methods,
            "Accuracy": [results["metrics"][m]["accuracy"] for m in methods],
            "Precision": [results["metrics"][m]["precision"] for m in methods],
            "Recall": [results["metrics"][m]["recall"] for m in methods],
            "F1 Score": [results["metrics"][m]["f1"] for m in methods]
        })
        
        # Create a bar chart for metrics comparison
        fig = px.bar(
            metrics_df.melt(id_vars=["Method"], var_name="Metric", value_name="Value"),
            x="Method", y="Value", color="Metric", barmode="group",
            title=f"Performance Metrics for {dataset_name}",
            labels={"Value": "Score", "Method": "Feature Selection Method"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics in a table
        st.dataframe(metrics_df.set_index("Method").round(4))
        
    with tabs[1]:  # Confusion Matrices
        st.header("Confusion Matrices")
        
        # Create 3 columns for confusion matrices
        cols = st.columns(3)
        
        for i, method in enumerate(methods):
            with cols[i]:
                cm = results["metrics"][method]["confusion_matrix"]
                
                # Plot confusion matrix
                fig = px.imshow(
                    cm, 
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Negative (0)", "Positive (1)"],
                    y=["Negative (0)", "Positive (1)"],
                    title=f"{method} Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display additional metrics
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                metrics_text = f"""
                - True Positives: {tp}
                - True Negatives: {tn}
                - False Positives: {fp}
                - False Negatives: {fn}
                - Specificity: {specificity:.4f}
                """
                st.markdown(metrics_text)
    
    with tabs[2]:  # ROC Curves
        st.header("ROC Curves Comparison")
        
        # Create ROC curves for all methods
        fig = go.Figure()
        
        for method in methods:
            y_test = results["data"]["y_test"]
            y_pred_proba = results["metrics"][method]["y_pred_proba"]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{method} (AUC = {roc_auc:.4f})',
                mode='lines'
            ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.7, y=0.05),
            width=800,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # Component Analysis
        st.header("Component Analysis")
        
        # Plot explained variance for PCA
        pca = results["components"]["PCA"]
        
        # Create tabs for PCA and ICA analysis
        component_tabs = st.tabs(["PCA Analysis", "ICA Analysis", "Feature Importance"])
        
        with component_tabs[0]:  # PCA Analysis
            # Plot explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(explained_variance) + 1)),
                    y=explained_variance,
                    name="Explained Variance"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumulative_variance) + 1)),
                    y=cumulative_variance,
                    name="Cumulative Variance",
                    mode="lines+markers"
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title="PCA Explained Variance",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance Ratio",
                legend=dict(x=0.7, y=0.05)
            )
            
            fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the first two principal components
            if results["data"]["X_train_pca"].shape[1] >= 2:
                fig = px.scatter(
                    x=results["data"]["X_train_pca"][:, 0],
                    y=results["data"]["X_train_pca"][:, 1],
                    color=results["data"]["y_train"],
                    title="First two Principal Components",
                    labels={"x": "PC1", "y": "PC2", "color": "Target"},
                    color_discrete_sequence=["blue", "red"]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with component_tabs[1]:  # ICA Analysis
            # Display the first two independent components
            if results["data"]["X_train_ica"].shape[1] >= 2:
                fig = px.scatter(
                    x=results["data"]["X_train_ica"][:, 0],
                    y=results["data"]["X_train_ica"][:, 1],
                    color=results["data"]["y_train"],
                    title="First two Independent Components",
                    labels={"x": "IC1", "y": "IC2", "color": "Target"},
                    color_discrete_sequence=["blue", "red"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyze kurtosis of independent components
                ica_components = results["data"]["X_train_ica"]
                kurtosis_values = []
                
                for i in range(ica_components.shape[1]):
                    # Calculate kurtosis
                    component = ica_components[:, i]
                    mean = np.mean(component)
                    std = np.std(component)
                    kurt = np.mean(((component - mean) / std) ** 4) - 3
                    kurtosis_values.append(kurt)
                
                # Plot kurtosis values
                fig = px.bar(
                    x=list(range(1, len(kurtosis_values) + 1)),
                    y=kurtosis_values,
                    title="Kurtosis of Independent Components",
                    labels={"x": "Component", "y": "Kurtosis"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Note:** Kurtosis measures the "tailedness" of the probability distribution.
                Higher kurtosis indicates more extreme outliers. ICA aims to find components with 
                non-Gaussian distributions, often characterized by higher kurtosis.
                """)
        
        with component_tabs[2]:  # Feature Importance
            # Show feature importance from the baseline model
            feature_importance = results["models"]["Baseline"].feature_importances_
            
            if "X_train" in results["data"]:
                if isinstance(results["data"]["X_train"], pd.DataFrame):
                    feature_names = results["data"]["X_train"].columns
                else:
                    try:
                        # Try to get original feature names
                        feature_names = X.columns
                    except:
                        feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
            else:
                feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
            
            # Create a DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title='Feature Importance from Random Forest',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation between features and PCA components
            if "PCA" in results["components"]:
                if hasattr(results["components"]["PCA"], "components_"):
                    pca_components = results["components"]["PCA"].components_
                    
                    if len(pca_components) > 0 and len(feature_names) == pca_components.shape[1]:
                        st.subheader("PCA Components Correlation with Original Features")
                        
                        # Create heatmap
                        component_names = [f"PC{i+1}" for i in range(len(pca_components))]
                        pca_df = pd.DataFrame(
                            pca_components,
                            columns=feature_names,
                            index=component_names
                        )
                        
                        fig = px.imshow(
                            pca_df,
                            labels=dict(x="Feature", y="Component", color="Correlation"),
                            title="PCA Components Correlation with Features",
                            color_continuous_scale='RdBu_r',
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:  # Time Comparison
        st.header("Computational Time Comparison")
        
        # Extract timing information
        time_df = pd.DataFrame({
            "Method": methods,
            "Transform Time (s)": [results["times"][m]["transform"] for m in methods],
            "Training Time (s)": [results["times"][m]["train"] for m in methods],
            "Total Time (s)": [results["times"][m]["total"] for m in methods]
        })
        
        # Create stacked bar chart for timing breakdown
        fig = px.bar(
            time_df.melt(id_vars=["Method"], value_vars=["Transform Time (s)", "Training Time (s)"], 
                         var_name="Time Component", value_name="Seconds"),
            x="Method", y="Seconds", color="Time Component", barmode="stack",
            title="Computational Time Breakdown",
            labels={"Method": "Feature Selection Method"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display timing in a table
        st.dataframe(time_df.set_index("Method").round(4))
        
        st.markdown("""
        The timing comparison shows:
        - **Transform Time**: Time taken to reduce dimensionality (PCA/ICA)
        - **Training Time**: Time taken to train the Random Forest classifier
        - **Total Time**: Combined time for both operations
        """)

# Main application flow
st.markdown("## Dataset Information")

# Initialize data containers
X = None
y = None
dataset_name = None

if uploaded_file is not None:
    # Read the data
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Data info
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Select target column
        target_col = st.selectbox(
            "Select Target Column",
            options=df.columns,
            index=len(df.columns)-1
        )
        
        # Preprocess data
        X, y = preprocess_data(df, target_col)
        dataset_name = uploaded_file.name
        
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        # Use default dataset as fallback
        X, y, dataset_name = load_default_dataset()
else:
    # Use default dataset
    X, y, dataset_name = load_default_dataset()
    st.write("Using default diabetes dataset from scikit-learn")
    
    # Show dataset information
    st.write(f"Shape: {X.shape[0]} rows, {X.shape[1]} columns")
    st.write("Features:", ", ".join(X.columns))
    
    # Display data preview
    df_preview = pd.concat([X.head(), pd.DataFrame(y.head(), columns=['Target'])], axis=1)
    st.dataframe(df_preview)

# Run analysis button
if st.button("Run Analysis"):
    # Check if data is available
    if X is not None and y is not None:
        with st.spinner("Running analysis... This may take a moment."):
            # Run the analysis
            results = run_analysis(X, y, dataset_name)
            
            # Display the results
            st.success("Analysis complete!")
            create_visualizations(results, dataset_name)
            
            # Display detailed classification reports
            with st.expander("Detailed Classification Reports"):
                for method in ["Baseline", "PCA", "ICA"]:
                    st.subheader(f"{method} Classification Report")
                    y_test = results["data"]["y_test"]
                    y_pred = results["metrics"][method]["y_pred"]
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4))
    else:
        st.error("No data available for analysis.")

# Add explanations
with st.expander("About the Methods"):
    st.markdown("""
    ## Principal Component Analysis (PCA)
    
    PCA is a dimensionality reduction technique that transforms the data into a new coordinate system
    where the greatest variance comes to lie on the first coordinate (first principal component), 
    the second greatest variance on the second coordinate, and so on.
    
    **Characteristics:**
    - Finds orthogonal directions (components) that maximize variance
    - Components are uncorrelated
    - Works best when features have linear relationships
    - Sensitive to feature scaling (requires standardization)
    
    ## Independent Component Analysis (ICA)
    
    ICA is a statistical technique for revealing hidden factors that underlie sets of random variables,
    measurements, or signals. ICA defines a generative model for observed multivariate data, 
    which is typically given as a large database of samples.
    
    **Characteristics:**
    - Finds statistically independent components
    - Assumes non-Gaussian distribution of sources
    - Often used for blind source separation
    - Can identify more complex relationships than PCA
    
    ## Random Forest Classifier
    
    Random Forest is an ensemble learning method that operates by constructing multiple decision trees
    during training time and outputting the class that is the mode of the classes output by individual trees.
    
    **Characteristics:**
    - Ensemble of decision trees
    - Robust to overfitting
    - Can handle high-dimensional data
    - Provides feature importance measures
    """)

# Footer
st.markdown("""
---
### How to use this application:
1. Upload your dataset (CSV format) or use the default diabetes dataset
2. Configure parameters in the sidebar
3. Click "Run Analysis" to see the comparison results
""")