# predictor.py
import joblib
import numpy as np

# Tải mô hình đã huấn luyện
model_data = joblib.load("water_prediction_model_20250731_140543.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

def predict_water(data: dict) -> float:
    """
    Dự đoán lượng nước tưới dựa trên dữ liệu đầu vào
    :param data: dict chứa các giá trị cảm biến
    :return: lượng nước (ml)
    """
    # Tạo input vector theo đúng thứ tự feature
    X = np.array([[data.get(col, 0) for col in feature_names]])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return float(y_pred[0])
