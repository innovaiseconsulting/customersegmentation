import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

# Generate features
age = np.random.randint(18, 70, n_customers)
income = np.random.normal(50000, 15000, n_customers)
spending_score = np.random.randint(1, 100, n_customers)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Annual Income': income,
    'Spending Score': spending_score
})

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Function to create scatter plot
def create_scatter_plot(x, y, hue):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=hue, data=df, palette='viridis')
    plt.title(f'{x} vs {y} by Cluster')
    return plt

# Streamlit app
st.title('Customer Segmentation Model')

st.write('This app demonstrates a simple customer segmentation model using K-means clustering.')

st.subheader('Dataset Overview')
st.write(df.head())

st.subheader('Cluster Statistics')
st.write(df.groupby('Cluster').mean())

st.subheader('Visualizations')
plot_x = st.selectbox('Select X-axis', ['Age', 'Annual Income', 'Spending Score'])
plot_y = st.selectbox('Select Y-axis', ['Age', 'Annual Income', 'Spending Score'])

if plot_x != plot_y:
    fig = create_scatter_plot(plot_x, plot_y, 'Cluster')
    st.pyplot(fig)
else:
    st.write('Please select different features for X and Y axes.')

st.subheader('Customer Segmentation Prediction')
st.write('Enter customer details to predict the segment:')

input_age = st.number_input('Age', min_value=18, max_value=70, value=30)
input_income = st.number_input('Annual Income', min_value=0, max_value=150000, value=50000)
input_spending = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)

if st.button('Predict Segment'):
    input_data = np.array([[input_age, input_income, input_spending]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    st.write(f'The customer belongs to Cluster {cluster}')

st.write('Note: This is a simplified model for demonstration purposes. Real-world segmentation would involve more features and sophisticated techniques.')

print("Streamlit app code generated successfully. You can now run this code in a Streamlit environment.")
