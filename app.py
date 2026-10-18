
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Public Transport Data Analytics",
    page_icon="🚇",
    layout="wide"
)

st.title("🚇 Public Transport Data Analytics")
st.subheader("Chennai Metro Passenger Demand Analysis")

st.write(
    "Analyze passenger boarding patterns and identify "
    "busy hours using Chennai Metro ridership data."
)


# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    file_path = "data/cmrl_hourly_ridership.csv"

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    return df


try:
    df = load_data()

    st.success("Hourly ridership dataset loaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

    # Hourly Passenger Demand
    st.subheader("Hourly Passenger Demand")

    hourly_data = df.groupby(
        "hour", as_index=False
    )["boardings"].mean()

    fig = px.line(
        hourly_data,
        x="hour",
        y="boardings",
        markers=True,
        title="Average Passenger Boardings by Hour",
        labels={
            "hour": "Hour of the Day",
            "boardings": "Average Boardings"
        }
    )

    st.plotly_chart(fig, width="stretch")

    # Date-wise Passenger Demand
    st.subheader("Date-wise Passenger Demand")

    daily_data = df.groupby(
        "date", as_index=False
    )["boardings"].sum()

    fig2 = px.line(
        daily_data,
        x="date",
        y="boardings",
        markers=True,
        title="Total Passenger Boardings by Date",
        labels={
            "date": "Date",
            "boardings": "Total Boardings"
        }
    )

    st.plotly_chart(fig2, width="stretch")

    # Peak Passenger Demand
    st.subheader("Peak Passenger Demand")

    peak_hour = hourly_data.loc[
        hourly_data["boardings"].idxmax()
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Peak Demand Hour",
            f"{int(peak_hour['hour'])}:00"
        )

    with col2:
        st.metric(
            "Average Boardings During Peak Hour",
            f"{peak_hour['boardings']:,.0f}"
        )

    # Dataset Summary
    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric(
            "Total Boardings",
            f"{df['boardings'].sum():,.0f}"
        )

    with col3:
        st.metric("Dates Available", df["date"].nunique())

except Exception as e:
    st.error(f"Unable to load dataset: {e}")


# ---------------- SPARK ML MODEL ----------------

@st.cache_resource
def train_prediction_model():

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("PassengerDemandDashboard")
        .master("local[2]")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    spark_df = spark.read.csv(
        "data/cmrl_hourly_ridership.csv",
        header=True,
        inferSchema=True
    )

    spark_df = spark_df.select(
        "hour", "boardings"
    ).dropna()

    assembler = VectorAssembler(
        inputCols=["hour"],
        outputCol="features"
    )

    training_data = assembler.transform(
        spark_df
    ).select("features", "boardings")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="boardings",
        maxIter=5
    )

    model = lr.fit(training_data)

    return spark, model


# ---------------- PASSENGER DEMAND PREDICTION ----------------

st.divider()
st.subheader("Passenger Demand Prediction")

st.write(
    "Select an hour to estimate passenger boardings "
    "using a basic machine learning model."
)

selected_hour = st.slider(
    "Select Hour of the Day",
    min_value=0,
    max_value=23,
    value=17,
    key="prediction_hour"
)

try:
    with st.spinner("Loading or training prediction model..."):
        spark, model = train_prediction_model()

    # Use the trained Spark Linear Regression model
    # coefficients for a quick prediction calculation.
    prediction_value = float(
        model.coefficients[0] * selected_hour
        + model.intercept
    )

    predicted_boardings = max(
        0,
        round(prediction_value)
    )

    st.metric(
        "Estimated Passenger Boardings",
        f"{predicted_boardings:,}"
    )

    st.caption(
        "This is a basic estimate based on hour only. "
        "It is not a guaranteed future passenger forecast."
    )

except Exception as e:
    st.error(
        "Prediction could not be generated. "
        "Check the VS Code terminal for details."
    )
    st.code(str(e))