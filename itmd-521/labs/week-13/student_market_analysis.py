# rachanapotha

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, year, month, avg, stddev
from pyspark.sql.types import IntegerType, FloatType

# Initialize Spark Session with Minio (S3-compatible storage) configuration
conf = {
    "spark.hadoop.fs.s3a.access.key": "minio",
    "spark.hadoop.fs.s3a.secret.key": "minio123",
    "spark.hadoop.fs.s3a.endpoint": "http://127.0.0.1:9000",
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem"
}

spark = SparkSession.builder \
    .appName("rachanapotha data transformation") \
    .config('spark.driver.host', 'spark-edge.service.consul') \
    .config(conf=conf) \
    .getOrCreate()

# Load the dataset
df = spark.read.text("s3a://itmd-521/80.txt")

# Transform dataset with specific schema
splitDF = df.withColumn('WeatherStation', df['_c0'].substr(5, 6)) \
    .withColumn('WBAN', df['_c0'].substr(11, 5)) \
    .withColumn('ObservationDate', to_date(df['_c0'].substr(16, 8), 'yyyyMMdd')) \
    .withColumn('ObservationHour', df['_c0'].substr(24, 4).cast(IntegerType())) \
    .withColumn('Latitude', df['_c0'].substr(29, 6).cast(FloatType()) / 1000) \
    .withColumn('Longitude', df['_c0'].substr(35, 7).cast(FloatType()) / 1000) \
    .withColumn('Elevation', df['_c0'].substr(47, 5).cast(IntegerType())) \
    .withColumn('WindDirection', df['_c0'].substr(61, 3).cast(IntegerType())) \
    .withColumn('WDQualityCode', df['_c0'].substr(64, 1).cast(IntegerType())) \
    .withColumn('SkyCeilingHeight', df['_c0'].substr(71, 5).cast(IntegerType())) \
    .withColumn('SCQualityCode', df['_c0'].substr(76, 1).cast(IntegerType())) \
    .withColumn('VisibilityDistance', df['_c0'].substr(79, 6).cast(IntegerType())) \
    .withColumn('VDQualityCode', df['_c0'].substr(86, 1).cast(IntegerType())) \
    .withColumn('AirTemperature', df['_c0'].substr(88, 5).cast(FloatType()) / 10) \
    .withColumn('ATQualityCode', df['_c0'].substr(93, 1).cast(IntegerType())) \
    .withColumn('DewPoint', df['_c0'].substr(94, 5).cast(FloatType()) / 10) \
    .withColumn('DPQualityCode', df['_c0'].substr(99, 1).cast(IntegerType())) \
    .withColumn('AtmosphericPressure', df['_c0'].substr(100, 5).cast(FloatType()) / 10) \
    .withColumn('APQualityCode', df['_c0'].substr(105, 1).cast(IntegerType())).drop('_c0')

# Save the cleaned data to Minio in different formats
splitDF.write.csv("s3a://itmd-521/80-uncompressed.csv", header=True)
splitDF.coalesce(1).write.csv("s3a://itmd-521/80.csv", header=True)
splitDF.write.csv("s3a://itmd-521/80-compressed.csv", header=True, compression="lz4")
splitDF.write.parquet("s3a://itmd-521/80.parquet")

# Calculate average temperature per month per year
monthly_avg_temp = splitDF.groupBy(year("ObservationDate").alias("Year"), month("ObservationDate").alias("Month")) \
    .agg(avg("AirTemperature").alias("AverageTemperature"))

# Calculate standard deviation of temperature per month over the decade
monthly_std_dev = monthly_avg_temp.groupBy("Month") \
    .agg(stddev("AverageTemperature").alias("TemperatureStdDev"))

# Save results
monthly_std_dev.write.parquet("s3a://itmd-521/part-three.parquet")

# Take only 12 records (one for each month) and write to CSV
monthly_std_dev.coalesce(1).write.csv("s3a://itmd-521/part-three.csv", header=True)

# Stop the Spark session
spark.stop()
