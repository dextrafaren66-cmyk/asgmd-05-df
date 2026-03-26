"""
Spaceship Titanic, Streamlit Prediction App
"""

import os
import joblib
import pandas as pd
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, os.pardir, "artifacts", "logistic_regression_pipeline.pkl")
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_single_input() -> pd.DataFrame:
    st.sidebar.header("Passenger Details")

    passenger_id = st.sidebar.text_input("PassengerId", value="9999_01")
    name = st.sidebar.text_input("Name", value="Test Passenger")

    home_planet = st.sidebar.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    destination = st.sidebar.selectbox(
        "Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]
    )

    cryo_sleep = st.sidebar.selectbox("CryoSleep", [False, True])
    vip = st.sidebar.selectbox("VIP", [False, True])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)

    deck = st.sidebar.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
    cabin_num = st.sidebar.number_input("Cabin Number", min_value=0, max_value=2000, value=0)
    side = st.sidebar.selectbox("Side", ["P", "S"])
    cabin = f"{deck}/{cabin_num}/{side}"

    st.sidebar.subheader("Spending")
    room_service = st.sidebar.number_input("RoomService", min_value=0.0, value=0.0)
    food_court = st.sidebar.number_input("FoodCourt", min_value=0.0, value=0.0)
    shopping_mall = st.sidebar.number_input("ShoppingMall", min_value=0.0, value=0.0)
    spa = st.sidebar.number_input("Spa", min_value=0.0, value=0.0)
    vr_deck = st.sidebar.number_input("VRDeck", min_value=0.0, value=0.0)

    data = {
        "PassengerId": [passenger_id],
        "HomePlanet": [home_planet],
        "CryoSleep": [cryo_sleep],
        "Cabin": [cabin],
        "Destination": [destination],
        "Age": [float(age)],
        "VIP": [vip],
        "RoomService": [room_service],
        "FoodCourt": [food_court],
        "ShoppingMall": [shopping_mall],
        "Spa": [spa],
        "VRDeck": [vr_deck],
        "Name": [name],
    }

    return pd.DataFrame(data)


def main():
    st.set_page_config(page_title="Spaceship Titanic", page_icon="🚀", layout="wide")
    st.title("🚀 Spaceship Titanic Predictor")

    pipe = load_model()

    tab_single, tab_batch = st.tabs(["Single Passenger", "Batch (CSV Upload)"])

    with tab_single:
        input_df = build_single_input()

        st.subheader("Input Data")
        st.dataframe(input_df, use_container_width=True)

        if st.button("Predict", type="primary"):
            prediction = pipe.predict(input_df)[0]
            proba = pipe.predict_proba(input_df)[0]

            if prediction == 1:
                st.success(f"Transported! (confidence: {proba[1]:.1%})")
            else:
                st.error(f"Not transported (confidence: {proba[0]:.1%})")

    with tab_batch:
        st.write("Upload a CSV with the same columns as train.csv (without Transported)")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Loaded {batch_df.shape[0]} rows")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Predict Batch", type="primary"):
                preds = pipe.predict(batch_df)
                probas = pipe.predict_proba(batch_df)[:, 1]

                results = batch_df[["PassengerId"]].copy()
                results["Transported"] = preds.astype(bool)
                results["Probability"] = probas.round(4)

                st.subheader("Predictions")
                st.dataframe(results, use_container_width=True)

                transported_count = preds.sum()
                total = len(preds)
                st.write(
                    f"{transported_count} out of {total} passengers predicted transported "
                    f"({transported_count / total:.1%})"
                )

                csv_output = results.to_csv(index=False)
                st.download_button(
                    "Download predictions as CSV",
                    csv_output,
                    "predictions.csv",
                    "text/csv",
                )


if __name__ == "__main__":
    main()
