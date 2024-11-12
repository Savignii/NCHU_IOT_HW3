import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Step 1: 生成400個隨機數，範圍為0到1000
np.random.seed(42)
X = np.random.randint(0, 1001, size=(400, 1))

# Step 2: 資料前處理，將介於400到700之間的label設為1，其餘設為0
labels = np.where((X >= 600) & (X <= 900), 1, 0)

# Step 3: 建立SVM和LR模型
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), labels, test_size=0.2)

# SVM模型
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# 邏輯回歸模型
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# 評估模型
svm_predictions = svm_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Function to display the classification report
def show_classification_report(predictions, y_true, model_name):
    st.subheader(f"{model_name} Classification Report")
    accuracy = accuracy_score(y_true, predictions)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.text(classification_report(y_true, predictions))

# Step 1: Generate 600 random points for the 3D classification problem
num_points = 600
mean = 3
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from the origin
distances = np.sqrt(x1**2 + x2**2)

# Assign labels Y=0 for points within distance 4, Y=1 for the rest
Y = np.where(distances < 4, 0, 1)

# Step 2: Calculate x3 as a Gaussian function of x1 and x2
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 3: Train a LinearSVC to find a separating hyperplane
X = np.column_stack((x1, x2, x3))
clf = LinearSVC()
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Streamlit Layout
st.title("Machine Learning Models Evaluation")

# Sidebar: Option to choose model
model_option = st.sidebar.selectbox("Choose Model", ["Logistic Regression vs SVM", "3D-plot SVM"])

# Step 4: Show Model Evaluation and Plot Based on Selection
if model_option == "Logistic Regression vs SVM":
    show_classification_report(lr_predictions, y_test, "Logistic Regression")
    
    # Plotting the results of Logistic Regression
    st.subheader("Logistic Regression Prediction Results")
    sorted_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sorted_indices]
    Y_test_sorted = y_test[sorted_indices]
    y1_sorted_lr = lr_predictions[sorted_indices]

    

    show_classification_report(svm_predictions, y_test, "SVM")
    
    # Plotting the results of SVM
    st.subheader("SVM Prediction Results")
    sorted_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sorted_indices]
    Y_test_sorted = y_test[sorted_indices]
    y1_sorted_svm = svm_predictions[sorted_indices]

    # Create plot
    plt.figure(figsize=(7, 6))
    plt.scatter(X_test_sorted, Y_test_sorted, color='gray', label='True Labels')
    plt.scatter(X_test_sorted, y1_sorted_lr, color='blue', marker='x', label='Logistic Regression')
    x_lr_range = np.linspace(0, 1000, 400)
    y_lr_predict = lr_model.predict_proba(x_lr_range.reshape(-1,1))[:, 1]
    plt.plot(x_lr_range, y_lr_predict, color='green', linewidth=2, label='Decision boundary', linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X vs Y and Logistic Regression Prediction')
    plt.legend()
    st.pyplot()

    # Create plot
    plt.figure(figsize=(7, 6))
    plt.scatter(X_test_sorted, Y_test_sorted, color='gray', label='True Labels')
    plt.scatter(X_test_sorted, y1_sorted_svm, color='red', marker='s', label='SVM')
    x_svm_range = np.linspace(0, 1000, 400)
    y_svm_predict = svm_model.predict_proba(x_svm_range.reshape(-1,1))[:, 1]
    plt.plot(x_svm_range, y_svm_predict, color='green', linewidth=2, label='Decision boundary', linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X vs Y and SVM Prediction')
    plt.legend()
    st.pyplot()

elif model_option == "3D-plot SVM":
    

    # Plotting the 3D separation using Plotly for interactivity
    st.subheader("3D Data and Separating Hyperplane")

    # Create 3D scatter plot using Plotly
    trace_0 = go.Scatter3d(
        x=x1[Y==0], y=x2[Y==0], z=x3[Y==0],
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Y=0'
    )
    trace_1 = go.Scatter3d(
        x=x1[Y==1], y=x2[Y==1], z=x3[Y==1],
        mode='markers',
        marker=dict(color='red', size=5),
        name='Y=1'
    )

    # Create mesh grid for the hyperplane
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    # Plot hyperplane surface
    trace_2 = go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='Blues', opacity=0.5,
        showscale=False
    )

    # Create layout with axis labels and title
    layout = go.Layout(
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3'
        ),
        title='3D Scatter Plot with Separating Hyperplane'
    )

    # Combine the traces into the plot
    fig = go.Figure(data=[trace_0, trace_1, trace_2], layout=layout)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
