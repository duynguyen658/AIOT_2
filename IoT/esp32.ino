#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// ==========================
// WiFi & MQTT cấu hình
// ==========================
const char* ssid = "Saveloka";
const char* password = "S79!@56k";

const char* mqtt_server = "192.168.1.207";  
int mqtt_port = 1883;

String device_id = "esp32_01";  
String mqtt_topic = "sensor/data/" + device_id;

// ==========================
// Cảm biến
// ==========================
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define SOIL_SENSOR_1 34
#define SOIL_SENSOR_2 35

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.printf("Kết nối WiFi: %s\n", ssid);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nĐã kết nối WiFi");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nKhông thể kết nối WiFi.");
  }
}

void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Đang kết nối MQTT...");
    if (client.connect(device_id.c_str())) {
      Serial.println("Đã kết nối MQTT");
    } else {
      Serial.printf("Thất bại, lỗi: %d. Thử lại sau 5s...\n", client.state());
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  if (!client.connected()) {
    reconnect_mqtt();
  }
  client.loop();

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int soil1 = analogRead(SOIL_SENSOR_1);
  int soil2 = analogRead(SOIL_SENSOR_2);

  int soil1_percent = map(soil1, 4095, 0, 0, 100);
  int soil2_percent = map(soil2, 4095, 0, 0, 100);
  int avg_soil = (soil1_percent + soil2_percent) / 2;

  if (!isnan(temperature) && !isnan(humidity)) {
    Serial.printf("🌡 %.1f°C | 💧 %.1f%% | 🌱 TB: %d%%\n", temperature, humidity, avg_soil);

    String payload = "{";
    payload += "\"device_id\":\"" + device_id + "\",";
    payload += "\"temperature\":" + String(temperature, 1) + ",";
    payload += "\"humidity\":" + String(humidity, 1) + ",";
    payload += "\"soil_avg\":" + String(avg_soil);
    payload += "}";

    client.publish(mqtt_topic.c_str(), payload.c_str());
  } else {
    Serial.println("Lỗi đọc cảm biến DHT11");
  }

  delay(5000);
}
