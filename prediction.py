import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Start Spark Session
spark = SparkSession.builder \
    .appName("PassengerDemandPrediction") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "data/cmrl_hourly_ridership.csv",
    header=True,
    inferSchema=True
)

# Select required columns
df = df.select("hour", "boardings").dropna()

# Prepare features
assembler = VectorAssembler(
    inputCols=["hour"],
    outputCol="features"
)

data = assembler.transform(df).select("features", "boardings")

# Train machine learning model
model = LinearRegression(
    featuresCol="features",
    labelCol="boardings"
)

trained_model = model.fit(data)

# Make predictions
predictions = trained_model.transform(data)

# Display predictions
predictions.select(
    "features",
    "boardings",
    "prediction"
).show(10, truncate=False)

spark.stop()