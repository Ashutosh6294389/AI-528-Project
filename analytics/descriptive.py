from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.format("delta").load("storage/delta_tables/youtube")

df.groupBy("category").avg("views").show()
df.groupBy("category").avg("engagement").show()