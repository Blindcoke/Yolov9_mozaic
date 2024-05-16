import cv2
import math
import numpy as np

from ultralytics import YOLO

class Face:
    def __init__(self, x1, y1, x2, y2, hp=0, scale=1.5):
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        self.scale = scale
        scaled_width = (x2 - x1) / 2 * scale
        scaled_height = (y2 - y1) / 2 * scale
        self.x1 = int(x_center - scaled_width)
        self.y1 = int(y_center - scaled_height)
        self.x2 = int(x_center + scaled_width)
        self.y2 = int(y_center + scaled_height)
        self.hp = hp

    def update(self, x1, y1, x2, y2, hp):
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        scaled_width = (x2 - x1) / 2 * self.scale
        scaled_height = (y2 - y1) / 2 * self.scale
        self.x1 = int(x_center - scaled_width)
        self.y1 = int(y_center - scaled_height)
        self.x2 = int(x_center + scaled_width)
        self.y2 = int(y_center + scaled_height)
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
        mask_blur_ksize = int(frame_height / 16)
        if mask_blur_ksize % 2 == 0:
            mask_blur_ksize += 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        results_generator = self.model.track(video_path, stream=True, persist=True, conf=0.01, tracker="face_tracker.yaml")

        face_book = FaceBook()
        window_name = "Frame"

        for result in results_generator:
            frame = result.orig_img
            blur = cv2.resize(frame, (0, 0), fx=1.0 / pixel_size, fy=1.0 / pixel_size)
            blur = cv2.resize(blur, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask[:, :] = 0  # Set all values to 0
            face_book.cleanup()

            if result.boxes and result.boxes.shape[0] != 0:
                boxes_numpy = result.boxes.data.cpu().numpy()
                for box in boxes_numpy:
                    face_book.register(*map(int, box[:4]), hp=max(1, int(fps)), delta=frame_height / 12)
            for face in face_book:
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            mask = cv2.GaussianBlur(mask, (mask_blur_ksize, mask_blur_ksize), 0)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask / 255.0
            frame = (frame * (1 - mask) + blur * mask).astype(np.uint8)
            output_video.write(frame)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        return output_path

face_mozaic = FaceMozaic()
output_video_path = face_mozaic.add_mozaic("videos/test2_short.mp4")
print("Video successfully saved to:", output_video_path)
