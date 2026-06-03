from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26s.pt")

    results = model.train(
        data="data_2.yaml",
        epochs=100,
        imgsz=640,
        batch=48,
        seed=20,
        deterministic=True,
        name="animals_dataset_2",    
    )