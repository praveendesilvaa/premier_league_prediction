import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load and preprocess the dataset
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Calculate rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x.reset_index(drop=True), cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Train and evaluate the model
train = matches_rolling[matches_rolling["date"] < '2022-01-01']
test = matches_rolling[matches_rolling["date"] > '2022-01-01']

predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Predict and evaluate
preds = rf.predict(test[predictors])
print("Confusion Matrix:")
print(confusion_matrix(test["target"], preds))
print("\nClassification Report:")
print(classification_report(test["target"], preds))

# Calculate and print accuracy
accuracy = accuracy_score(test["target"], preds)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model ton] a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)
