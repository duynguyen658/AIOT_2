# database.py

import json
import os
from datetime import datetime
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/data/#"

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "YFx9lM4jzlKrJ_NNWSEL8iA9mzV0X7PsNY9TrNtmPfx4K43BZBwSeuzf-thUZ0mJeTwWE23BoLtEyTr_-8zDXg==")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "EG Group")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "IoT")

influx_client = InfluxDBClient(
    url=INFLUXDB_URL,
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG
)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)

        device_id = data.get("device_id")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        soil_avg = data.get("soil_avg")

        print(f"[{datetime.now()}] Received data from {device_id}: {data}")

        # Tạo điểm dữ liệu cho InfluxDB
        point = (
            Point("sensor_data")
            .tag("device_id", device_id)
            .field("temperature", float(temperature))
            .field("humidity", float(humidity))
            .field("soil_avg", float(soil_avg))
            .time(datetime.utcnow(), WritePrecision.NS)
        )

        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)

    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {e}")
        print(f"Payload nhận được: {msg.payload}")

def main():
    print("Khởi động backend - Đợi dữ liệu từ MQTT...")

    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
    mqtt_client.subscribe(MQTT_TOPIC)

    mqtt_client.loop_forever()

if __name__ == "__main__":
    main()
