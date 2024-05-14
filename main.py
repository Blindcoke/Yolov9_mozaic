import cv2
import math

from ultralytics import YOLO

class Face:
    def __init__(self, x1, y1, x2, y2, hp=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.hp = hp

    def update(self, x1, y1, x2, y2, hp):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.hp = hp

    def distance(self, other):
        return math.dist([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2], [(other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2])

class FaceBook:
    def __init__(self):
        self.registry = []

    def register(self, x1, y1, x2, y2, hp, delta):
        face = Face(x1, y1, x2, y2)
        closest_face = None
        closest_distance = float("inf")
        for neighbour in self.registry:
            distance = face.distance(neighbour)
            if distance < delta and distance < closest_distance:
                closest_face = neighbour
                closest_distance = distance
        if closest_face:
            closest_face.update(x1, y1, x2, y2, hp)
        else:
            self.registry.append(Face(x1, y1, x2, y2, hp))

    def cleanup(self):
        for face in self.registry:
            face.hp -= 1
        self.registry = [face for face in self.registry if face.hp > 0]

    def __getitem__(self, index):
        return self.registry[index]
    
    def __iter__(self):
        for face in self.registry:
            yield face

class FaceMozaic:
    def __init__(self, model_path="yolo/yolov8n-face.pt"):
        self.model = YOLO(model_path)
        self.pixel_size_rel = 0.015

    def add_mozaic(self, video_path):
        output_path=video_path.split(".")[0] + "_mozaic.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{frame_width}x{frame_height} @ {fps} FPS")

        pixel_size = int(frame_height * self.pixel_size_rel)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        results_generator = self.model.predict(video_path, stream=True)

        face_book = FaceBook()

        for result in results_generator:
            frame = result.orig_img
            face_book.cleanup()

            if result.boxes and result.boxes.shape[0] != 0:
                boxes = result.boxes[0].data.cpu().numpy() 
                for box in boxes:
                    face_book.register(*map(int, box[:4]), hp=min(1, int(fps / 3)), delta=frame_height / 12)
            for face in face_book:
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                face = frame[y1:y2, x1:x2]
                face_width = x2 - x1
                face_height = y2 - y1
                face = cv2.resize(face, (0, 0), fx=1.0 / pixel_size, fy=1.0 / pixel_size)
                face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = face

            output_video.write(frame)

        cap.release()
        output_video.release()
        return output_path

face_mozaic = FaceMozaic()
output_video_path = face_mozaic.add_mozaic("videos/test2_short.mp4")
print("Video successfully saved to:", output_video_path)