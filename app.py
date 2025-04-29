import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from io import StringIO

# 1. Data Preprocessing
# Assuming the data is provided as a string in the document
data_str = """states,cities,population,violent_crime,murder,rape,robbery,agrv_assault,prop_crime,burglary,larceny,vehicle_theft
Pennsylvania,"Abington Township, Montgomery County","55,731",197.4,1.8,14.4,70,111.2,1979.1,296.1,1650.8,32.3
Oregon,Albany,"51,084",86.1,0,19.6,45,21.5,3092.9,438.5,2470.4,184
Louisiana,Alexandria,"48,449",1682.2,18.6,28.9,293.1,1341.6,7492.4,2010.4,5102.3,379.8
... (rest of your data) ..."""

# Convert string to DataFrame
df = pd.read_csv(StringIO(data_str))

# Clean the data
df = df.replace('#N/A', np.nan)  # Replace #N/A with NaN
df = df.dropna()  # Drop rows with missing values for simplicity

# Convert population and crime rates to numeric (remove commas and convert)
df['population'] = df['population'].str.replace(',', '').astype(float)
for col in ['violent_crime', 'murder', 'rape', 'robbery', 'agrv_assault', 'prop_crime', 'burglary', 'larceny', 'vehicle_theft']:
    df[col] = df[col].astype(float)

# Create a binary target variable (e.g., violent crime > 250)
df['high_crime'] = (df['violent_crime'] > 250).astype(int)

# Features and target
X = df[['population', 'murder', 'rape', 'robbery', 'agrv_assault', 'prop_crime', 'burglary', 'larceny', 'vehicle_theft']]
y = df['high_crime']

# 2. Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('crime_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# 3. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 4. Streamlit App
st.title("Crime Prediction Web App")

st.write("""
### Enter City Crime Data to Predict High Crime Risk
Input the population and crime statistics per 100,000 people.
""")

# Input fields
population = st.number_input("Population", min_value=0, value=50000)
murder = st.number_input("Murder Rate", min_value=0.0, value=0.0)
rape = st.number_input("Rape Rate", min_value=0.0, value=0.0)
robbery = st.number_input("Robbery Rate", min_value=0.0, value=0.0)
agrv_assault = st.number_input("Aggravated Assault Rate", min_value=0.0, value=0.0)
prop_crime = st.number_input("Property Crime Rate", min_value=0.0, value=0.0)
burglary = st.number_input("Burglary Rate", min_value=0.0, value=0.0)
larceny = st.number_input("Larceny Rate", min_value=0.0, value=0.0)
vehicle_theft = st.number_input("Vehicle Theft Rate", min_value=0.0, value=0.0)

# Prediction
if st.button("Predict"):
    # Load the model
    with open('crime_prediction_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Prepare input data
    input_data = np.array([[population, murder, rape, robbery, agrv_assault, prop_crime, burglary, larceny, vehicle_theft]])
    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data)

    st.write(f"**Prediction:** {'High Crime Risk' if prediction[0] == 1 else 'Low Crime Risk'}")
    st.write(f"**Probability of High Crime:** {probability[0][1]:.2f}")

    # Display metrics
    st.write("### Model Evaluation Metrics (on Test Set)")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

# Note: For a production app, consider adding data validation, scaling features, and handling edge cases.
