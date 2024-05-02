from time import time

import cv2
import numpy as np
import supervision as sv
import torch

from ultralytics import YOLO

from utils import check_collision, get_person_bbox


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model("model/best.pt")
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.DEFAULT, thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self, model_path):
       
        model = YOLO(model_path)
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
         # Extract detections for person class
        try:
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                class_id = boxes.cls[0]

                if class_id == 0.0:
            
                    xyxys.append(result.boxes.xyxy.cpu().numpy())
                    confidences.append(result.boxes.conf.cpu().numpy())
                    class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        except IndexError:
            print("Nothing detected")    
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _
                in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        shopping_cart = []
        while True:
            start_time = time()
            
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            try:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                print("BOXES:", boxes)
                print("LABELS:", classes)
                print("CLASS_DICT:", self.model.model.names)

                person_bbox, boxes, classes = get_person_bbox(boxes, classes)
                print("PERSON:", person_bbox)
                print("NEW BOXES:", boxes)
                print("NEW CLASSES:", classes)
                
            except IndexError:
                print("Nothing detected")            
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        


def main():

    detector = ObjectDetection(capture_index=0)
    detector()


if __name__ == "__main__":
    main()