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

        import matplotlib.pyplot as plt
import seaborn as sns

st.header("IPL Visual Insights")

matches_df = pd.read_csv("data/matches.csv")

# 1. Matches by city
st.subheader("Matches Played by City")
city_counts = match_df['city'].dropna().value_counts().head(10)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x=city_counts.values, y=city_counts.index, ax=ax1, palette="viridis")
ax1.set_xlabel("Number of Matches")
ax1.set_ylabel("City")
ax1.set_title("Top 10 Cities by Matches Hosted")
st.pyplot(fig1)

# 2. Toss decision distribution
st.subheader("Toss Decision Distribution")
toss_counts = match_df['toss_decision'].value_counts()

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.pie(toss_counts.values, labels=toss_counts.index, autopct="%1.1f%%", startangle=90)
ax2.set_title("Bat vs Field Decisions After Toss")
st.pyplot(fig2)

# 3. Top winning teams
st.subheader("Top Winning Teams")
winner_counts = match_df['winner'].dropna().value_counts().head(10)

fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.barplot(x=winner_counts.index, y=winner_counts.values, ax=ax3, palette="magma")
ax3.set_xlabel("Team")
ax3.set_ylabel("Wins")
ax3.set_title("Top 10 Teams by Wins")
plt.xticks(rotation=45)
st.pyplot(fig3)