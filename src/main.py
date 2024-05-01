import cv2


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        if not ret:
            break

        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model("src/test.png")
    print(results) 


if __name__ == "__main__":
    main()