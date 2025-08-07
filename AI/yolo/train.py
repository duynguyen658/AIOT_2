from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolo11s.pt")

    # Train
    results = model.train(
        data="data.yaml",
        epochs=10,
        batch=8,
        imgsz=416,
        device=0
    )

if __name__ == '__main__':
    main()
