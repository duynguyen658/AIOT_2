import numpy as np
import pandas as pd
from AI import EnhancedWaterPredictionModel
import matplotlib.pyplot as plt
# Load mô hình đã huấn luyện
model = EnhancedWaterPredictionModel()
model.load_model("water_prediction_model_20250731_140543.pkl")

# Tạo dữ liệu đầu vào giả lập (đa dạng tình huống)
test_data = pd.DataFrame([
    {'soil_moisture': 100, 'air_humidity': 0, 'air_temp': 0, 'light': 100, 'rainfall': 0},
    {'soil_moisture': 100, 'air_humidity': 70, 'air_temp': 30, 'light': 500, 'rainfall': 5},
    {'soil_moisture': 60, 'air_humidity': 40, 'air_temp': 15, 'light': 200, 'rainfall': 0},
    {'soil_moisture': 40, 'air_humidity': 20, 'air_temp': 10, 'light': 100, 'rainfall': 0},
    {'soil_moisture': 100, 'air_humidity': 90, 'air_temp': 35, 'light': 900, 'rainfall': 15},
    {'soil_moisture': 30, 'air_humidity': 50, 'air_temp': 25, 'light': 600, 'rainfall': 0},
    {'soil_moisture': 80, 'air_humidity': 10, 'air_temp': 5, 'light': 300, 'rainfall': 3},
])

# Tạo đặc trưng như lúc huấn luyện
test_data['moisture_humidity_ratio'] = test_data['soil_moisture'] / (test_data['air_humidity'] + 1)
test_data['temp_humidity_interaction'] = test_data['air_temp'] * test_data['air_humidity']
X_test = test_data[model.feature_names]
predictions = model.predict(X_test.values)

print("\nDự đoán lượng nước tưới (lít):")
for i, pred in enumerate(predictions):
    print(f"Input {i+1}: {X_test.iloc[i].values} => {pred:.2f} lít")
importances = model.model.feature_importances_
plt.barh(model.feature_names, importances)
plt.title("Feature Importance")
plt.show()
# Kiểm tra độ thay đổi đầu ra
print("\nĐộ chênh lệch đầu ra:", max(predictions) - min(predictions))
