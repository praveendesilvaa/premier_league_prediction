import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])

# Encode categorical features
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

# Handle 'Away' and similar strings in other columns
def clean_non_numeric_values(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the specific columns
columns_to_clean = ["venue_code", "opp_code", "hour", "day_code"]
matches = clean_non_numeric_values(matches, columns_to_clean)

# Rolling averages calculation
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

# Convert all predictors to numeric and drop rows with non-numeric values
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
matches_rolling[predictors] = matches_rolling[predictors].apply(pd.to_numeric, errors='coerce')
matches_rolling = matches_rolling.dropna(subset=predictors)

# Train the model
train = matches_rolling[matches_rolling["date"] < '2022-01-01']
test = matches_rolling[matches_rolling["date"] > '2022-01-01']

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Define function to preprocess input for prediction
def preprocess_input(team1, team2, df):
    teams = df["team"].unique()
    venues = df["venue"].unique()
    
    # Encode teams and venues
    le_team = LabelEncoder().fit(teams)
    le_venue = LabelEncoder().fit(venues)
    
    if team1 not in le_team.classes_ or team2 not in le_team.classes_:
        raise ValueError("One of the teams is not recognized.")
    
    team1_code = le_team.transform([team1])[0]
    team2_code = le_team.transform([team2])[0]
    
    # Handle venue encoding
    venue_code = le_venue.transform(['Neutral'])[0] if 'Neutral' in le_venue.classes_ else le_venue.classes_[0]
    
    # Get rolling averages for the teams
    def get_team_rolling_averages(team):
        team_data = df[(df['team'] == team) & (df['date'] < '2022-01-01')]
        if team_data.empty:
            raise ValueError(f"No data available for team: {team}")
        averages = team_data[new_cols].mean().fillna(0)
        return averages.tolist()
    
    rolling_averages_team1 = get_team_rolling_averages(team1)
    rolling_averages_team2 = get_team_rolling_averages(team2)
    
    # Construct predictors
    predictors_team1 = [float(venue_code), float(team2_code)] + [float(x) for x in rolling_averages_team1]
    predictors_team2 = [float(venue_code), float(team1_code)] + [float(x) for x in rolling_averages_team2]
    
    # Print predictors for debugging
    print(f"Predictors for team1: {predictors_team1}")
    print(f"Predictors for team2: {predictors_team2}")
    
    return predictors_team1, predictors_team2

# Define function to predict match winner
def predict_match_winner(team1, team2, df, model):
    predictors_team1, predictors_team2 = preprocess_input(team1, team2, df)
    
    # Ensure predictors are numeric
    predictors_team1 = [float(x) for x in predictors_team1]
    predictors_team2 = [float(x) for x in predictors_team2]
    
    # Print processed predictors for debugging
    print(f"Processed predictors for team1: {predictors_team1}")
    print(f"Processed predictors for team2: {predictors_team2}")
    
    prediction_team1_win = model.predict_proba([predictors_team1])[0][1]
    prediction_team2_win = model.predict_proba([predictors_team2])[0][1]
    
    if prediction_team1_win > prediction_team2_win:
        return f"{team1} is more likely to win."
    elif prediction_team1_win < prediction_team2_win:
        return f"{team2} is more likely to win."
    else:
        return "The match is likely to be a draw."

# Main function
def main():
    df = matches_rolling
    print("Available teams:")
    teams = df['team'].unique()
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    
    team1_idx = int(input("Enter the number corresponding to the first team: ")) - 1
    team2_idx = int(input("Enter the number corresponding to the second team: ")) - 1
    
    team1 = teams[team1_idx]
    team2 = teams[team2_idx]
    
    try:
        result = predict_match_winner(team1, team2, df, rf)
        print(result)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
