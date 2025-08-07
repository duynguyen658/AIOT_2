#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "Saveloka";
const char* password = "S79!@56k";
const char* mqtt_server = "192.168.1.207";  

WiFiClient espClient;
PubSubClient client(espClient);

const int relayPin = D4;  // Chân kết nối relay
const float FLOW_RATE_ML_PER_SEC = 0.05; // tốc độ bơm (ml/s) => 0.05ml/s

void setup_wifi() {
  delay(10);
  Serial.begin(115200);
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived: ");
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  StaticJsonDocument<128> doc;
  DeserializationError error = deserializeJson(doc, message);
  if (error) {
    Serial.println("Failed to parse JSON");
    return;
  }

  if (doc.containsKey("water")) {
    float water_amount = doc["water"];  // đơn vị: ml
    Serial.printf("Nhận lệnh tưới: %.1f ml\n", water_amount);

    float duration_sec = water_amount / FLOW_RATE_ML_PER_SEC;
    Serial.printf("Bật bơm trong %.1f giây\n", duration_sec);

    digitalWrite(relayPin, HIGH);  // Bật bơm
    delay(duration_sec * 1000);    // Chờ đủ thời gian
    digitalWrite(relayPin, LOW);   // Tắt bơm
  }
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP8266_Watering")) {
      client.subscribe("esp8266/watering");
      Serial.println("Đã kết nối MQTT và subscribe esp8266/watering");
    } else {
      Serial.print(".");
      delay(1000);
    }
  }
}

void setup() {
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
