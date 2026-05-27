from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="animals_v1",    
    )
    print(results.save_dir)

if __name__ == "__main__":
    main()

main
