from ultralytics import YOLO

if __name__ == "__main__":
#def main():
    model = YOLO("./runs/detect/animals_dataset_1-6/weights/best.pt")

    results = model.val(
    data="data_1.yaml",
    split="test",  
    )

    print(results.box.map)
