import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the diabetes dataset
data = pd.read_csv('/diabetes_data.csv')  # Replace with your actual dataset path

# 2. Prepare the data
# Assuming 'Outcome' column contains diabetes information (0 or 1)
# If the column is named differently, change it accordingly
# data['target'] = data['Outcome']  # This line is removed, we already have 'Outcome'
# Splitting data into features (X) and target (y)
# Changing the drop column to 'Outcome' as that is the actual column
X = data.drop(columns='Outcome')  
y = data['Outcome']  # Assigning 'Outcome' column as the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = LogisticRegression(max_iter=1000)  # Initialize the model
model.fit(X_train, y_train)  # Train the model on training data

# 4. Make predictions for multiple inputs
new_data = pd.DataFrame({  # Create a DataFrame for new data
    'Pregnancies': [0, 8, 8, 13, 2],
    'Glucose': [148, 183, 183, 150, 155],
    'BloodPressure': [72, 64, 64, 0, 76],
    'SkinThickness': [35, 0, 0, 32, 30],
    'Insulin': [0, 0, 0, 84, 0],
    'BMI': [33.6, 23.3, 23.3, 45.8, 30.1],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.514, 0.38],
    'Age': [50, 32, 32, 27, 60]
})

predictions = model.predict(new_data)  # Make predictions on the new data

# 5. Display predictions
for i, prediction in enumerate(predictions):
    print(f"Person {i + 1}: Prediction = {prediction} (1 = Diabetes Risk, 0 = Normal)")

