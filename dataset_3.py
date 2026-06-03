from ultralytics import YOLO

if __name__ == "__main__":
    ### loading of pretrained model
    model = YOLO("yolo26s.pt")

    ### executing model traning with selected settings
    results = model.train(
        data="data_3.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        seed=45,
        deterministic=True,
        name="animals_dataset_3",    
    )