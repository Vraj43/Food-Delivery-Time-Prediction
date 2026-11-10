
import os
import joblib
import pandas as pd
import streamlit as st

from src.features import make_features, FEATURES

MODEL_PATH = "models/delivery_time_model.joblib"

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="centered",
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🍔 Food Delivery Time Prediction")
st.write(
    "Estimate how long a food delivery may take "
    "using machine learning."
)

if model is None:
    st.error("Trained model not found. Please run python train_model.py first.")
    st.stop()

st.info("Enter the delivery details below and click Predict.")

with st.form("prediction_form"):
    st.subheader("Delivery Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Delivery Partner Age",
            min_value=18,
            max_value=60,
            value=28,
        )

        rating = st.number_input(
            "Delivery Partner Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.5,
            step=0.1,
        )

        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            options=[0, 1, 2],
            index=1,
        )

        distance = st.number_input(
            "Delivery Distance (km)",
            min_value=1.4,
            max_value=20.9,
            value=9.2,
            step=0.5,
        )

        deliveries = st.selectbox(
            "Number of Deliveries",
            options=[0, 1, 2, 3,
                     4],
            index=1,
        )

    with col2:
        weather = st.selectbox(
            "Weather",
            ["Sunny", "Stormy", "Sandstorms",
             "Cloudy", "Windy", "Fog"],
        )

        traffic = st.selectbox(
            "Road Traffic",
            ["Low", "Medium", "High", "Jam"],
        )

        order_type = st.selectbox(
            "Order Type",
            ["Snack", "Meal", "Drinks", "Buffet"],
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["motorcycle", "scooter", "electric_scooter"],
        )

        festival = st.selectbox(
            "Festival",
            ["No", "Yes"],
        )

        city = st.selectbox(
            "City Type",
            ["Metropolitian", "Urban", "Semi-Urban"],
        )

    st.subheader("Order Timing")

    order_date = st.date_input("Order Date")
    order_time = st.time_input(
        "Order Time",
        value=pd.to_datetime("18:00").time(),
    )
    pickup_time = st.time_input(
        "Pickup Time",
        value=pd.to_datetime("18:10").time(),
    )

    submitted = st.form_submit_button(
        "Predict Delivery Time",
        use_container_width=True,
    )


if submitted:
    # Prepare the input using the same feature function
    # used during model training.
    input_df = pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": deliveries,
        "distance_km": distance,
        "Order_Date": order_date.strftime("%d-%m-%Y"),
        "Time_Orderd": order_time.strftime("%H:%M"),
        "Time_Order_picked": pickup_time.strftime("%H:%M"),
        "Weather_conditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city,
    }])

    features = make_features(input_df)
    prediction = model.predict(features)[0]
    prediction = max(1, round(float(prediction), 1))

    st.success("Prediction completed!")
    st.metric(
        label="Estimated Delivery Time",
        value=f"{prediction} minutes",
    )

    st.caption(
        "This is a machine learning estimate, not a guaranteed delivery time."
    )
