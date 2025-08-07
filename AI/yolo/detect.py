from ultralytics import YOLO
import os
import cv2

# Đường dẫn đến model đã huấn luyện
model_path = "D:/AIOT/admin/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Thư mục chứa ảnh đầu vào
input_folder = "D:/AIOT/admin/testimage/train/Potato___Late_blight"
output_folder = "D:/AIOT/admin/output"

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua từng ảnh trong thư mục
for file_name in os.listdir(input_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, file_name)
        img = cv2.imread(image_path)

        # Dự đoán
        results = model.predict(source=img, save=False, conf=0.5)

        # Lấy ảnh có bounding boxes
        for r in results:
            annotated_img = r.plot()

        # Lưu ảnh kết quả
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, annotated_img)

print(f" Dự đoán xong. Ảnh đã lưu vào: {output_folder}")
