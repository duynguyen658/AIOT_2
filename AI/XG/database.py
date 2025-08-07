import json
import time
from paho.mqtt import client as mqtt_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
# Cấu hình MQTT
broker = '192.168.1.207'
port = 1883
topic = "esp32/sensor_1"
client_id = f'python-mqtt-{time.time()}'

# Cấu hình InfluxDB
influx_url = "http://localhost:8086"
influx_token = "YFx9lM4jzlKrJ_NNWSEL8iA9mzV0X7PsNY9TrNtmPfx4K43BZBwSeuzf-thUZ0mJeTwWE23BoLtEyTr_-8zDXg=="
influx_org = "EG Group"
influx_bucket = "IoT"

# Kết nối InfluxDB
influx_client = InfluxDBClient(
    url=influx_url,
    token=influx_token,
    org=influx_org
)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Hàm xử lý khi nhận MQTT message
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        temp = float(data.get("temp", 0))
        hum = float(data.get("hum", 0))
        soil = float(data.get("soil", 0))

        point = Point("iot_sensor") \
            .field("temperature", temp) \
            .field("humidity", hum) \
            .field("soil_moisture", soil) \
            .time(time.time_ns(), WritePrecision.NS)

        write_api.write(bucket=influx_bucket, org=influx_org, record=point)

        print(f"Ghi InfluxDB: temp={temp}, hum={hum}, soil={soil}")

    except Exception as e:
        print(f"Lỗi xử lý dữ liệu: {e}")


# Kết nối MQTT
def connect_mqtt():
    client = mqtt_client.Client(client_id)
    client.on_connect = lambda client, userdata, flags, rc: print("🔌 MQTT đã kết nối" if rc == 0 else f" MQTT lỗi: {rc}")
    client.on_message = on_message
    client.connect(broker, port)
    return client

# Main
def run():
    client = connect_mqtt()
    client.subscribe(topic)
    client.loop_forever()

if __name__ == '__main__':
    run()
