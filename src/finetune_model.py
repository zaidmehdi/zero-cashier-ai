from ultralytics import YOLO


def main():

    model = YOLO('model/yolov8n.pt')
    model.train(data='datasets/data/dataset.yaml', epochs=1000, batch=8, patience=100)


if __name__ == "__main__":
    main()