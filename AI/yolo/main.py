from ultralytics import YOLO
import cv2

# Load mô hình huấn luyện của bạn
model = YOLO("D:/AIOT/admin/runs/detect/train/weights/best.pt")

# Mở webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.4)

    # Vẽ kết quả lên ảnh
    annotated_frame = results[0].plot()

    # Hiển thị ảnh
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
