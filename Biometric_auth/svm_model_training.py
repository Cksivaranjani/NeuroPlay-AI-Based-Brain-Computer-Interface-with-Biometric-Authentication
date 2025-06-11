import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('dataset/mental-state.csv')  # Adjust the path to your dataset

print(data.columns)
print(data.head(10))

# Separate features (all columns except 'Label') and labels ('Label')
X = data.drop('Label', axis=1)
y = data['Label']

# Normalize the features (scaling is important if the model was trained on scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # tune parameters
svm.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# See unique labels and their counts
print("Unique label values:", data['Label'].unique())
print("\nLabel counts:\n", data['Label'].value_counts())

# Save the trained model and scaler to disk
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Manual testing (input manually entered frequency values)
# These should be similar to the features used during training, including all frequency bands
# Convert manual input to DataFrame with the correct column names
manual_input_1 = pd.DataFrame([X.iloc[0].values], columns=X.columns)  # Convert to DataFrame with column names
manual_input_2 = pd.DataFrame([X.iloc[3].values], columns=X.columns)  # Convert to DataFrame with column names
manual_input_3 = pd.DataFrame([X.iloc[7].values], columns=X.columns)  # Convert to DataFrame with column names

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Standardize the manual inputs (no warning about feature names)
manual_inputs = [manual_input_1, manual_input_2, manual_input_3]
scaled_inputs = [scaler.transform(input_data) for input_data in manual_inputs]

# Load the saved model
svm_model = joblib.load('svm_model.pkl')

# Define a dictionary to map predictions to readable states
state_map = {2.0: "Concentrated", 1.0: "Stressed", 0.0: "Relaxed or Idle"}

# Loop through each input and make predictions
for i, input_data in enumerate(scaled_inputs, 1):
    predicted_state = svm_model.predict(input_data)
    print(f"\nPredicted mental state for input {i}: {predicted_state[0]}")
    print(f"The mental state is: {state_map.get(predicted_state[0], 'Unknown state')}")
