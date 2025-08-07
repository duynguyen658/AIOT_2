## Hệ thống AIoT cho Nông nghiệp Thông minh

Giới thiệu
Hệ thống AIoT này được phát triển nhằm tự động hóa và tối ưu hóa việc tưới tiêu, giám sát môi trường và nhận diện bệnh cây trong nông nghiệp. Hệ thống tích hợp các thành phần AI (dự đoán tưới nước, nhận diện bệnh), Backend (xử lý dữ liệu, giao tiếp MQTT, lưu trữ InfluxDB, giao diện web) và IoT (ESP32/ESP8266 thu thập dữ liệu, điều khiển bơm).

## Thành phần chính
- **AI**: Dự đoán lượng nước tưới bằng XGBoost, nhận diện bệnh cây bằng YOLO.
- **Backend**: Xử lý dữ liệu cảm biến, giao tiếp AI và IoT qua MQTT, lưu trữ dữ liệu với InfluxDB, cung cấp giao diện web.
- **IoT**: ESP32 thu thập dữ liệu môi trường, ESP8266 điều khiển bơm tưới qua MQTT.

## Hướng dẫn cài đặt
### 1. Cài đặt trên Windows
- Cài Python >= 3.8
- Cài đặt các thư viện cần thiết (xem phần Yêu cầu phần mềm)
- Chạy các thành phần Backend và AI bằng lệnh `python main.py` hoặc tương ứng trong từng thư mục.

### 2. Cài đặt bằng Docker
- Cài Docker Desktop
- Di chuyển vào thư mục `Backend/`
- Chạy lệnh:
  ```bash
  docker-compose up -d
  ```
- InfluxDB sẽ chạy ở cổng 8086.

## Yêu cầu phần mềm
- Python >= 3.8
- pip, venv
- Các thư viện: xgboost, pandas, numpy, scikit-learn, matplotlib, seaborn, flask, paho-mqtt, influxdb-client, joblib, ...
- Phần cứng: ESP32, ESP8266, cảm biến nhiệt độ/độ ẩm/độ ẩm đất, relay, bơm nước

## Hướng dẫn sử dụng
- **AI**: Chạy các script trong `AI/XG/` để huấn luyện và dự đoán lượng nước tưới. Chạy `AI/yolo/` để nhận diện bệnh cây từ ảnh.
- **Backend**: Chạy `main.py` để nhận dữ liệu cảm biến và dự đoán tưới, chạy `web_server.py` để khởi động giao diện web.
- **IoT**: Nạp mã nguồn `esp32.ino` cho ESP32 (thu thập dữ liệu), `relayeg.ino` cho ESP8266 (điều khiển bơm).
- Truy cập giao diện web tại `http://localhost:5000` (mặc định).

## Thông tin tác giả
- Tên: Hồ Minh Duy
- Email: nguyenphanhongduy658@gmail.com
