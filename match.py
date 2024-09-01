#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd

# Load the dataset from a CSV file
matches = pd.read_csv("matches.csv", index_col=0)

# Display the first few rows of the DataFrame
matches.head()

# Display the shape of the DataFrame (number of rows and columns)
matches.shape

# Count the occurrences of each team in the 'team' column
matches["team"].value_counts()

# Display all rows where the team is 'Liverpool'
matches[matches["team"] == "Liverpool"]

# Count the occurrences of each unique value in the 'round' column
matches["round"].value_counts()

# Check the data types of each column in the DataFrame
matches.dtypes

# Convert the 'date' column to datetime format
matches["date"] = pd.to_datetime(matches["date"])

# Verify the conversion by checking the data types again
matches.dtypes

# Convert the 'venue' column to categorical type and then to integer codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes

# Convert the 'opponent' column to categorical type and then to integer codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

# Display the DataFrame to see the changes made
matches

# Extract the hour part from the 'time' column and convert it to integer
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

# Display the DataFrame to see the changes made
matches

# Extract the day of the week from the 'date' column and store it in 'day_code'
matches["day_code"] = matches["date"].dt.dayofweek

# Display the DataFrame to see the changes made
matches

# Create a binary target column where 'W' (win) is 1 and others are 0
matches["target"] = (matches["result"] == "W").astype("int")

# Display the DataFrame to see the changes made
matches

# Import the RandomForestClassifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Split the data into training and testing sets based on the 'date' column
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']

# Define the predictors (independent variables)
predictors = ["venue_code", "opp_code", "hour", "day_code"]

# Train the RandomForest model using the training data
rf.fit(train[predictors], train["target"])

# Predict the target variable on the test data
preds = rf.predict(test[predictors])

# Import the accuracy_score function to evaluate the model
from sklearn.metrics import accuracy_score

# Calculate the accuracy of the model on the test data
error = accuracy_score(test["target"], preds)

# Output the accuracy score
error

# Create a DataFrame to compare actual vs. predicted results
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))

# Create a confusion matrix to evaluate the predictions
pd.crosstab(index=combined["actual"], columns=combined["predicted"])

# Import precision_score to evaluate the model's precision
from sklearn.metrics import precision_score

# Calculate the precision of the model on the test data
precision_score(test["target"], preds)

# Group the matches DataFrame by 'team'
grouped_matches = matches.groupby("team")

# Retrieve and sort the matches for 'Manchester City' by date
group = grouped_matches.get_group("Manchester City").sort_values("date")

# Display the sorted DataFrame for 'Manchester City'
group

# Define a function to calculate rolling averages for a given group
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")  # Sort the group by date
    rolling_stats = group[cols].rolling(3, closed='left').mean()  # Calculate rolling means with a window of 3
    group[new_cols] = rolling_stats  # Add the rolling averages to the group
    group = group.dropna(subset=new_cols)  # Drop rows with NaN values in the new columns
    return group

# Define columns to calculate rolling averages on
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]

# Generate new column names for rolling averages
new_cols = [f"{c}_rolling" for c in cols]

# Apply the rolling averages function to the 'Manchester City' group
rolling_averages(group, cols, new_cols)

# Apply the rolling averages function to each team group and concatenate the results
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x.reset_index(drop=True), cols, new_cols))

# Flatten the multi-level index created by the groupby operation
matches_rolling = matches_rolling.droplevel('team')

# Reset the index of the DataFrame to a sequential integer index
matches_rolling.index = range(matches_rolling.shape[0])

# Display the updated DataFrame with rolling averages
matches_rolling

# Define a function to train the model and make predictions
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']  # Training data
    test = data[data["date"] > '2022-01-01']  # Testing data
    rf.fit(train[predictors], train["target"])  # Train the model
    preds = rf.predict(test[predictors])  # Make predictions
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)  # Compare actual vs. predicted
    error = precision_score(test["target"], preds)  # Calculate precision
    return combined, error

# Make predictions using the model and calculate the error
combined, error = make_predictions(matches_rolling, predictors + new_cols)

# Output the precision score
error

# Display the combined DataFrame with actual and predicted results
combined

# Merge the combined DataFrame with the original matches_rolling DataFrame for additional context
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Display the merged DataFrame
combined

# Define a class that returns the key if the key is missing in the dictionary
class MissingDict(dict):
    __missing__ = lambda self, key: key

# Create a mapping dictionary for team names to their common abbreviations
map_values = {"Brighton and Hove Albion": "Brighton", 
              "Manchester United": "Manchester Utd", 
              "Newcastle United": "Newcastle Utd", 
              "Tottenham Hotspur": "Tottenham", 
              "West Ham United": "West Ham", 
              "Wolverhampton Wanderers": "Wolves"}

# Create an instance of MissingDict using map_values
mapping = MissingDict(**map_values)

# Map the 'team' column to its abbreviated form using the mapping dictionary
combined["new_team"] = combined["team"].map(mapping)

# Display the DataFrame after mapping team names
combined

# Merge the combined DataFrame with itself on 'date' and 'new_team' to compare actual and predicted results
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# Display the merged DataFrame to analyze predictions
merged

# Count the actual outcomes where the first model predicted a win (1) and the second model predicted a loss (0)
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()

# Check the column names in the matches DataFrame
matches.columns
