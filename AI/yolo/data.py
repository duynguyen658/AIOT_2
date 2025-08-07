from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DHT Analysis").getOrCreate()
df = spark.read.csv("iot_telemetry_data.csv", header=True, inferSchema=True)

# Hiển thị schema và 5 dòng đầu
df.printSchema()
df.show(1000)

# Thống kê
df.describe(['temp', 'humidity']).show()

# Gắn nhãn (sử dụng when/otherwise)
from pyspark.sql.functions import when

df = df.withColumn("label", when((df.temp > 27.5) & (df.humidity > 70), "Hot_Humid")
                              .when((df.temp < 25) & (df.humidity < 50), "Cool_Dry")
                              .otherwise("Moderate"))

df.select("temp", "humidity", "label").show(100)
