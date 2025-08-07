import paho.mqtt.client as mqtt
from pymongo import MongoClient
import json
from datetime import datetime

# MongoDB setup
mongo_client = MongoClient("mongodb+srv://toanchuong94:kvn7b34DAGZ0PnDo@cluster0.avpca3p.mongodb.net/duy")
db = mongo_client["duy"]
collection = db["IoT"]

# MQTT setup
mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883
mqtt_topic = "esp8266/temp"

def on_connect(client, userdata, flags, rc):
    print("Đã kết nối MQTT với mã:", rc)
    client.subscribe(mqtt_topic)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temperature = data.get("temp", None)
        if temperature is not None:
            record = {
                "temperature": temperature,
                "timestamp": datetime.now()
            }
            collection.insert_one(record)
            print("Đã lưu vào MongoDB:", record)
        else:
            print("Không có nhiệt độ trong dữ liệu:", data)
    except Exception as e:
        print("Lỗi xử lý message:", e)

# Khởi tạo MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Kết nối MQTT
client.connect(mqtt_broker, mqtt_port, 60)

# Vòng lặp chính
client.loop_forever()
