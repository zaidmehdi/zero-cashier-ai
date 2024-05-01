from ultralytics import YOLO


def main():

    model = YOLO('model/yolov8n.pt')
    results = model.train(data='datasets/data/dataset.yaml', epochs=2)


if __name__ == "__main__":
    main()