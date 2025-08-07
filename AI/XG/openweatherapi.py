import requests
import json

API_KEY = "92c640ca500884a284ec054797d2fb98"  # Thay bằng API key của bạn
CITY_ID = 1566083


def fetch_and_save_weather(city_id, api_key, filename="weather_data.json"):
    url = f"http://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Ghi dữ liệu vào file JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Dữ liệu thời tiết đã lưu vào '{filename}'")
        return data

    except requests.RequestException as e:
        print(f"Lỗi khi gọi API: {e}")
        return None


# Gọi hàm
weather_data = fetch_and_save_weather(CITY_ID, API_KEY)
