from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model and data for preprocessing
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load and preprocess the dataset for encoding and rolling averages
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Define rolling average calculation function
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")  # Sort by date
    rolling_stats = group[cols].rolling(3, closed='left').mean()  # Calculate rolling means
    group[new_cols] = rolling_stats  # Add rolling averages
    group = group.dropna(subset=new_cols)  # Drop rows with NaNs
    return group

# Define columns for rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to the dataset
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x.reset_index(drop=True), cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# Create mapping for team names
map_values = {
    "Brighton and Hove Albion": "Brighton", 
    "Manchester United": "Manchester Utd", 
    "Newcastle United": "Newcastle Utd", 
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = pd.Series(map_values).to_dict()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    team1 = data.get('team1')
    team2 = data.get('team2')

    if team1 not in matches['team'].values or team2 not in matches['team'].values:
        return jsonify({'error': 'One or both team names are not in the dataset.'}), 400

    # Prepare the input features for both teams
    team1_data = matches_rolling[matches_rolling['team'] == team1].iloc[-1]
    team2_data = matches_rolling[matches_rolling['team'] == team2].iloc[-1]

    # Prepare the feature for prediction
    input_data_team1 = {
        'venue_code': matches['venue_code'].mode()[0],  # Example venue code, modify as needed
        'opp_code': matches['opp_code'].mode()[0],  # Example opponent code, modify as needed
        'hour': 15,  # Example hour
        'day_code': 2,  # Example day code
    }

    input_data_team2 = {
        'venue_code': matches['venue_code'].mode()[0],  # Example venue code, modify as needed
        'opp_code': matches['opp_code'].mode()[0],  # Example opponent code, modify as needed
        'hour': 15,  # Example hour
        'day_code': 2,  # Example day code
    }

    for col in new_cols:
        # Average rolling stats of the two teams
        input_data_team1[col] = team1_data[col]
        input_data_team2[col] = team2_data[col]

    input_df_team1 = pd.DataFrame([input_data_team1])
    input_df_team2 = pd.DataFrame([input_data_team2])

    # Predict.
    prediction_team1 = model.predict(input_df_team1)
    prediction_team2 = model.predict(input_df_team2)

    # Determine the winning team.
    if prediction_team1[0] == 1 and prediction_team2[0] == 0:
        result = team1
    elif prediction_team1[0] == 0 and prediction_team2[0] == 1:
        result = team2
    else:
        result = "Draw or Uncertain"

    return jsonify({'winning_team': result})


if __name__ == '__main__':
    app.run(debug=True)