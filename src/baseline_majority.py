import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_json("data/raw/IMDB_reviews.json", lines=True)  

# TEMP split of the dataset into training and testing sets
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=99, stratify=df["is_spoiler"]
)

# Find majority class in training data
majority_class = train_df["is_spoiler"].mode()[0]
print(f"Majority class: {majority_class}")

# Predict majority class for every test row
y_true = test_df["is_spoiler"]
y_pred = [majority_class] * len(y_true)

# Metrics
def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

results = evaluate_predictions(y_true, y_pred)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")