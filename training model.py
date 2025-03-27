import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=["label"])
y = df["label"]

# Encode the target variable
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))

# Save the trained model and encoder
with open("crop_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(y_encoder, encoder_file)

print("Model and encoder saved successfully!")
