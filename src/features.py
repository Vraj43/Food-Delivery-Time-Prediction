
import pandas as pd

TARGET = "Time_taken (min)"

NUMERICAL_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "distance_km",
    "order_year",
    "order_month",
    "order_day",
    "order_weekday",
    "order_hour",
    "pickup_hour",
    "prep_minutes",
]

CATEGORICAL_FEATURES = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def parse_time(series):
    """Convert time strings into datetime values."""
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="%H:%M",
        errors="coerce",
    )


def make_features(df):
    """Create model-ready features from the original dataset."""
    df = df.copy()

    # Convert date into useful numerical features.
    date = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    df["order_year"] = date.dt.year
    df["order_month"] = date.dt.month
    df["order_day"] = date.dt.day
    df["order_weekday"] = date.dt.dayofweek

    # Extract order and pickup hours.
    order_time = parse_time(df["Time_Orderd"])
    pickup_time = parse_time(df["Time_Order_picked"])

    df["order_hour"] = order_time.dt.hour
    df["pickup_hour"] = pickup_time.dt.hour

    # Calculate preparation time in minutes.
    prep_minutes = (
        (pickup_time - order_time).dt.total_seconds() / 60
    )

    # Account for orders picked up after midnight.
    prep_minutes = prep_minutes.where(
        prep_minutes >= 0,
        prep_minutes + 24 * 60,
    )

    df["prep_minutes"] = prep_minutes

    # Return only features used for prediction.
    return df[FEATURES]
