from flask import Flask, render_template, request, jsonify
from influxdb_client import InfluxDBClient
import paho.mqtt.publish as publish
import os

app = Flask(__name__)

# Cấu hình InfluxDB
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "YFx9lM4jzlKrJ_NNWSEL8iA9mzV0X7PsNY9TrNtmPfx4K43BZBwSeuzf-thUZ0mJeTwWE23BoLtEyTr_-8zDXg==")
ORG = "EG Group"
BUCKET = "IoT"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORG)

# Đọc dữ liệu mới nhất
def get_latest_sensor_data():
    query = f'''
    from(bucket:"{BUCKET}")
    |> range(start: -1h)
    |> filter(fn: (r) => r._measurement == "sensor_data")
    |> last()
    '''
    tables = client.query_api().query(query)
    result = {}
    for table in tables:
        for row in table.records:
            result[row.get_field()] = row.get_value()
    return result

@app.route('/')
def index():
    data = get_latest_sensor_data()
    return render_template('index.html', data=data)

@app.route('/pump', methods=['POST'])
def control_pump():
    amount = request.json.get('water')
    # Gửi MQTT điều khiển ESP8266
    publish.single(
        topic="esp8266/watering",
        payload=f'{{"water": {amount}}}',
        hostname="localhost",
        port=1883
    )
    return jsonify({"status": "ok", "water": amount})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
