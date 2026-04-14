from pyspark.sql.functions import col

def clean_data(df):
    return df \
        .withColumn("views", col("views").cast("int")) \
        .withColumn("likes", col("likes").cast("int")) \
        .withColumn("comments", col("comments").cast("int")) \
        .withColumn("engagement", col("likes") / col("views"))