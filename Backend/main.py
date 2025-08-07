from paho.mqtt import client as mqtt_client
import json
from predictor import predict_water

BROKER = "192.168.1.207"
PORT = 1883
TOPIC_SUB = "sensor/data/esp32_01"
TOPIC_PUB = "esp8266/watering"
CLIENT_ID = "ai_water_predictor"

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(" Dữ liệu nhận:", data)

        # Dự đoán
        water = predict_water(data)
        print(f" Gợi ý tưới: {water:.1f} l")

        # Gửi kết quả lại
        result = json.dumps({"water": round(water)})
        client.publish(TOPIC_PUB, result)
        print(" Đã gửi:", result)

    except Exception as e:
        print(" Lỗi:", e)

def start_mqtt():
    client = mqtt_client.Client(CLIENT_ID)
    client.connect(BROKER, PORT)
    client.subscribe(TOPIC_SUB)
    client.on_message = on_message
    print("MQTT đang chạy...")
    client.loop_forever()

if __name__ == "__main__":
    start_mqtt()
