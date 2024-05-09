import cv2
from ultralytics import YOLO

class FaceMozaic:
    def __init__(self, model_path="yolo/yolov8n-face.pt"):
        self.model = YOLO(model_path)

    def add_mozaic(self, video_path):
        output_path=video_path.split(".")[0] + "_mozaic.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        results_generator = self.model.predict(video_path, stream=True)

        for result in results_generator:
            frame = result.orig_img  
            if result.boxes:  
                boxes = result.boxes[0].data.cpu().numpy() 
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    face = frame[y1:y2, x1:x2]
                    face = cv2.resize(face, (0, 0), fx=0.05, fy=0.05)
                    face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = face

            output_video.write(frame)

        cap.release()
        output_video.release()
        return output_path

face_mozaic = FaceMozaic()
output_video_path = face_mozaic.add_mozaic("videos/test2_short.mp4")
print("Video successfully saved to:", output_video_path)
