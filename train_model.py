import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Read CSV files
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# 2. Clean team names
matches.replace("Delhi Daredevils", "Delhi Capitals", inplace=True)
matches.replace("Deccan Chargers", "Sunrisers Hyderabad", inplace=True)

# 3. Keep only useful columns
data = matches[["team1", "team2", "city", "toss_winner", "winner"]].dropna()

# 4. Create target column
# team1_win = 1 if team1 won, else 0
data["team1_win"] = (data["winner"] == data["team1"]).astype(int)

# 5. Inputs and output
X = data[["team1", "team2", "city", "toss_winner"]]
y = data["team1_win"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Preprocessing
categorical_features = ["team1", "team2", "city", "toss_winner"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 8. Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 9. Train model
model.fit(X_train, y_train)

# 10. Test model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 11. Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nmodel.pkl saved successfully")