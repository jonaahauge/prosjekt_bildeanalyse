from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")

    results = model.train(
        data="data_3.yaml",
        epochs=100,
        imgsz=640,
        batch=48,
        seed=30,
        deterministic=True,
        name="animals_dataset_3",    
    )