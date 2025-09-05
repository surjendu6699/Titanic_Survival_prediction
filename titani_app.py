import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pipeline model
model = joblib.load(r"C:\Users\surje\titanic_logistic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details to predict survival:")

# === User Inputs ===
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard (Parch)", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
cabin_deck = st.selectbox("Cabin Deck", ["None", "A", "B", "C", "D", "E", "F", "G", "T"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Other"])

# === Feature Engineering (replicate training) ===
# Encode Sex
sex = 1 if sex == "Male" else 0

# Encode Embarked (C is baseline)
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Family size and alone flag
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# AgeGroup (simple binning, same as training)
if age <= 12:
    age_group = 0
elif age <= 18:
    age_group = 1
elif age <= 35:
    age_group = 2
elif age <= 50:
    age_group = 3
else:
    age_group = 4

# Fare binning (similar logic as training)
if fare <= 7.91:
    fare_bin = 0
elif fare <= 14.454:
    fare_bin = 1
elif fare <= 31:
    fare_bin = 2
else:
    fare_bin = 3

# HasCabin flag
has_cabin = 0 if cabin_deck == "None" else 1

# CabinDeck one-hot
cabin_decks = {d: 0 for d in ["A","B","C","D","E","F","G","T"]}
if cabin_deck in cabin_decks:
    cabin_decks[cabin_deck] = 1

# Title one-hot
titles = {t: 0 for t in ["col","countess","don","dr","jonkheer","lady","major",
                         "master","me","miss","mlle","mr","mrs","ms","rev","sir"]}
title_key = title.lower()
if title_key in titles:
    titles[title_key] = 1
else:
    titles["other"] = 1  # catch-all

# === Build feature row matching training columns ===
features = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "AgeGroup": age_group,
    "HasCabin": has_cabin,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "FareBin": fare_bin,
    "Embarked_q": embarked_q,
    "Embarked_s": embarked_s,
    **{f"CabinDeck_{k.lower()}": v for k, v in cabin_decks.items()},
    **{f"Title_{k}": v for k, v in titles.items()}
}])

# === Predict ===
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f" Passenger is predicted to SURVIVE (Probability: {probability:.2f})")
    else:
        st.error(f"Passenger is predicted to NOT SURVIVE (Probability: {probability:.2f})")
