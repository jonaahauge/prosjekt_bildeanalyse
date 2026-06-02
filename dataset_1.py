from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=10,
        seed=10,
        deterministic=True,
        name="animals_dataset_1",    
    )
    print(results.save_dir)

if __name__ == "__main__":
    main()