import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data
# -----------------------------
training = pd.read_csv("Training.csv")
description = pd.read_csv("description.csv")
diets = pd.read_csv("diets.csv")
medications = pd.read_csv("medications.csv")
workout = pd.read_csv("workout_df.csv")

# Clean column names
training.columns = training.columns.str.strip()
description.columns = description.columns.str.strip()
diets.columns = diets.columns.str.strip()
medications.columns = medications.columns.str.strip()
workout.columns = workout.columns.str.strip()

# -----------------------------
# Prepare Model Data
# -----------------------------
X = training.drop("prognosis", axis=1)
y = training["prognosis"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 AI Medical Recommendation System")

st.write("Select your symptoms below:")

selected_symptoms = st.multiselect(
    "Choose Symptoms",
    X.columns
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:

        input_vector = np.zeros(len(X.columns))

        for symptom in selected_symptoms:
            index = X.columns.get_loc(symptom)
            input_vector[index] = 1

        similarity = cosine_similarity([input_vector], X)
        top_indices = similarity[0].argsort()[-3:][::-1]

        st.subheader("🔍 Top 3 Possible Diseases")

        for i in top_indices:
            st.write(f"**{y.iloc[i]}**  (Score: {round(similarity[0][i],3)})")

        best_disease = y.iloc[top_indices[0]]

        # -----------------------------
        # Description
        # -----------------------------
        st.subheader("📖 Description")

        desc = description[description["Disease"] == best_disease]["Description"].values
        if len(desc) > 0:
            st.write(desc[0])
        else:
            st.write("No description available.")

        # -----------------------------
        # Diet
        # -----------------------------
        st.subheader("🥗 Recommended Diet")

        diet_plan = diets[diets["Disease"] == best_disease]
        if not diet_plan.empty:
            st.write(diet_plan.iloc[0].values[1])
        else:
            st.write("No diet information available.")

        # -----------------------------
        # Medications
        # -----------------------------
        st.subheader("💊 Medications")

        meds = medications[medications["Disease"] == best_disease]
        if not meds.empty:
            st.write(meds.iloc[0].values[1])
        else:
            st.write("No medication information available.")

        # -----------------------------
        # Workout
        # -----------------------------
        st.subheader("Suggestions")
        st.write("Disease:", best_disease)

        work = workout[workout["disease"] == best_disease]
        if not work.empty:
            st.write(work.iloc[0].values[1])
        else:
            st.write("No information available")

        st.warning("⚠ this is not a real medical prediction it was concerened with small input only so make sure to consult doctor!")