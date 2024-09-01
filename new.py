#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report

# Load the dataset from a CSV file
matches = pd.read_csv("matches.csv", index_col=0)

# Convert the 'date' column to datetime format
matches["date"] = pd.to_datetime(matches["date"])

# Convert categorical columns to integer codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Extract hour and day of week from 'time' and 'date' columns
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# Create a binary target column where 'W' (win) is 1 and others are 0
matches["target"] = (matches["result"] == "W").astype("int")

# Define the RandomForestClassifier model
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=1)

# Define predictors
predictors = ["venue_code", "opp_code", "hour", "day_code"]
# Define rolling average columns
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to the dataset
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    rolling_stats.columns = new_cols  # Rename columns for rolling stats
    group = pd.concat([group, rolling_stats], axis=1)
    group = group.dropna(subset=new_cols)
    return group

matches_rolling = matches.groupby("team", group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols)).reset_index(drop=True)

# Train-test split based on date
train = matches_rolling[matches_rolling["date"] < '2022-01-01']
test = matches_rolling[matches_rolling["date"] > '2022-01-01']

# Debug: Check the shapes of the data
print("Train Data Shape:", train.shape)
print("Test Data Shape:", test.shape)

# Train the Randomorest model
def train_model(data):
    train_data = data[data["date"] < '2022-01-01']
    if not train_data.empty:
        rf.fit(train_data[predictors + new_cols], train_data["target"])
    else:
        print("No data available for training.")

# Define a function to make predictions
def make_predictions(data):
    test_data = data[data["date"] > '2022-01-01']
    if test_data.empty:
        print("No data available for testing.")
        return pd.DataFrame(), 0
    preds = rf.predict(test_data[predictors + new_cols])
    combined = pd.DataFrame({'actual': test_data["target"], 'predicted': preds}, index=test_data.index)
    error = precision_score(test_data["target"], preds)
    print(classification_report(test_data["target"], preds))
    return combined, error

# Define a function to predict match outcome between two teams
def predict_winner(team1, team2):
    if team1 not in matches["team"].unique() or team2 not in matches["team"].unique():
        return "One or both teams not found in the dataset."

    team1_data = matches[matches["team"] == team1]
    team2_data = matches[matches["team"] == team2]

    # Apply rolling averages to team data
    team1_data = rolling_averages(team1_data, cols, new_cols)
    team2_data = rolling_averages(team2_data, cols, new_cols)

    if team1_data.empty or team2_data.empty:
        return "Insufficient data for prediction."

    # Calculate mean features for prediction
    team1_mean_features = team1_data[predictors + new_cols].mean().values.reshape(1, -1)
    team2_mean_features = team2_data[predictors + new_cols].mean().values.reshape(1, -1)

    # Ensure the model is trained with the correct features
    print("Predictors used for training:", predictors + new_cols)

    # Train the model
    train_model(matches_rolling)

    # Predict using the trained model
    team1_prediction = rf.predict(team1_mean_features)[0]
    team2_prediction = rf.predict(team2_mean_features)[0]

    print(f"Team 1 prediction: {team1_prediction}")
    print(f"Team 2 prediction: {team2_prediction}")

    # Determine the predicted winner
    if team1_prediction > team2_prediction:
        return f"{team1} is predicted to win"
    elif team2_prediction > team1_prediction:
        return f"{team2} is predicted to win"
    else:
        return "It's a draw"

# Example usage
team1 = input("Enter the first team: ").strip()
team2 = input("Enter the second team: ").strip()
print(predict_winner(team1, team2))
