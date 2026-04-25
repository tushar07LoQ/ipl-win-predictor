import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data for dropdown values
matches = pd.read_csv("data/matches.csv")

# Clean old team names
matches.replace("Delhi Daredevils", "Delhi Capitals", inplace=True)
matches.replace("Deccan Chargers", "Sunrisers Hyderabad", inplace=True)

teams = sorted(list(set(matches["team1"]).union(set(matches["team2"]))))
cities = sorted(matches["city"].dropna().unique())

# Title
st.title("IPL Match Winner Predictor")
st.write("Choose match details and click predict.")

# Inputs
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
city = st.selectbox("Select City", cities)
toss_winner = st.selectbox("Select Toss Winner", [team1, team2])

# Button
if st.button("Predict Winner"):
    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be same")
    else:
        input_df = pd.DataFrame({
            "team1": [team1],
            "team2": [team2],
            "city": [city],
            "toss_winner": [toss_winner]
        })

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.success(f"{team1} is more likely to win")
        else:
            st.success(f"{team2} is more likely to win")

        st.write(f"{team1} win probability: {probability[1] * 100:.2f}%")
        st.write(f"{team2} win probability: {probability[0] * 100:.2f}%")