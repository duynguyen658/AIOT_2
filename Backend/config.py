import os
from dataclasses import dataclass


@dataclass
class Config:
    # MQTT Configuration
    MQTT_BROKER: str = "192.168.1.207"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""

    # Flask Configuration
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = True

    # InfluxDB Configuration
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = "YFx9lM4jzlKrJ_NNWSEL8iA9mzV0X7PsNY9TrNtmPfx4K43BZBwSeuzf-thUZ0mJeTwWE23BoLtEyTr_-8zDXg=="
    INFLUXDB_ORG: str = "EG Group"
    INFLUXDB_BUCKET: str = "IoT"

    # MQTT Topics
    TOPIC_SENSOR_DATA: str = "esp32/sensor_1"
    TOPIC_WATERING: str = "esp8266/watering"
    TOPIC_STATUS: str = "esp8266/status"
    TOPIC_CONTROL: str = "esp8266/control"


config = Config()