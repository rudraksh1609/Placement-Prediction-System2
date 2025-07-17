import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Placement_Prediction_data.csv")
df.drop(["Unnamed: 0", "StudentId"], axis=1, inplace=True)

# Encode categorical variables
label = LabelEncoder()
df["Internship"] = label.fit_transform(df["Internship"])
df["Hackathon"] = label.fit_transform(df["Hackathon"])
df["PlacementStatus"] = label.fit_transform(df["PlacementStatus"])

# Features and target
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base models (without deprecated parameter)
xgb = XGBClassifier(eval_metric='logloss', random_state=42, learning_rate=0.07, n_estimators=150)
gbc = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
logreg = LogisticRegression(max_iter=1000)

# Voting Classifier
ensemble = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('gbc', gbc),
    ('lr', logreg)
], voting='soft')

# Train ensemble
ensemble.fit(X_train, y_train)

# Save model
joblib.dump(ensemble, "placement_model.pkl")
