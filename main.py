import cv2
import math
import numpy as np
import sys
from matplotlib import pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

class Face:
    def __init__(self, x1, y1, x2, y2, hp=0, scale=1.15, fps=30, damping_factor=5, debug=False):
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
        self.alpha = 1.0
        self.fps = fps
        self.damping_factor = damping_factor / fps
        self.debug = debug
        if self.debug:
            self.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    def update(self, x1, y1, x2, y2, hp):
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        scaled_width = (x2 - x1) / 2 * self.scale
        scaled_height = (y2 - y1) / 2 * self.scale
        new_x1 = int(x_center - scaled_width)
        new_y1 = int(y_center - scaled_height)
        new_x2 = int(x_center + scaled_width)
        new_y2 = int(y_center + scaled_height)
        if new_x1 <= self.x1:
            self.x1 = new_x1
        else:
            self.x1 = int(self.x1 - (self.x1 - new_x1) * self.damping_factor)
        if new_y1 <= self.y1:
            self.y1 = new_y1
        else:
            self.y1 = int(self.y1 - (self.y1 - new_y1) * self.damping_factor)
        if new_x2 >= self.x2:
            self.x2 = new_x2
        else:
            self.x2 = int(self.x2 + (new_x2 - self.x2) * self.damping_factor)
        if new_y2 >= self.y2:
            self.y2 = new_y2
        else:
            self.y2 = int(self.y2 + (new_y2 - self.y2) * self.damping_factor)
        self.hp = hp

    def distance(self, other):
        return math.dist([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2], [(other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2])

class FaceBook:
    def __init__(self):
        self.registry = {}
        self.max_id = 0

    def register(self, x1, y1, x2, y2, hp, delta, fps=30, id=None):
        face = Face(x1, y1, x2, y2, hp, fps=fps)
        if id:
            if id in self.registry.keys():
                self.registry[id].update(x1, y1, x2, y2, hp)
            else:
                self.registry[id] = face
        else:
            closest_face = None
            closest_distance = float("inf")
            for neighbour in self.registry.values():
                distance = face.distance(neighbour)
                if distance < delta and distance < closest_distance:
                    closest_face = neighbour
                    closest_distance = distance
            if closest_face:
                closest_face.update(x1, y1, x2, y2, hp)
            else:
                self.registry[self.max_id] = face
                self.max_id += 1

    def contains(self, id):
        return id in self.registry.keys()

    def cleanup(self):
        for face in self.registry.values():
            if face.hp == 0:
                face.alpha -= 1.0 / face.fps
            else:
                face.hp -= 1
        self.registry = {k: v for k, v in self.registry.items() if v.alpha > 0}

    def __getitem__(self, index):
        return self.registry[index]
    
    def __iter__(self):
        for face in sorted(self.registry.values(), key=lambda face: face.alpha):
            yield face

class FaceMozaic:
    def __init__(self, model_path="yolo/yolov9c_best.pt", model_path2="yolo/yolov8n-face.pt"):
        self.model = YOLO(model_path)
        self.model2 = YOLO(model_path2)
        self.pixel_size_rel = 0.015
    
    def enlarge_box(self, box, frame_width, frame_height, scale=1.3):
        x1, y1, x2, y2 = box[:4]
        width = x2 - x1
        height = y2 - y1

        hw = width * scale / 2
        hh = height * scale / 2

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        x1_new = max(0, cx - hw)
        y1_new = max(0, cy - hh)
        x2_new = min(frame_width - 1, cx + hw)
        y2_new = min(frame_height - 1, cy + hh)

        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

    def add_mozaic(self, video_path):
        output_path=video_path.split(".")[0] + "_mozaic.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        buffer_size = int(fps * 0.6)
        print(f"{frame_width}x{frame_height} @ {fps} FPS")

        pixel_size = int(frame_height * self.pixel_size_rel)
        mask_blur_ksize = int(frame_height / 16)
        if mask_blur_ksize % 2 == 0:
            mask_blur_ksize += 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        results_generator = self.model.track(video_path, stream=True, show=True, persist=True, iou=0.05,conf=0.01, tracker="face_tracker.yaml")

        face_book = FaceBook()
        window_name = "Frame"
        buffer = []

        # box_tracker = defaultdict(int)
        box_threshold = frame_height * 20
        result = next(results_generator, None)
        while result is not None:
            result_boxes = result.boxes.data.cpu().numpy()
            for box in result_boxes:
                box_id = int(box[4])
                box_size = (box[2] - box[0]) * (box[3] - box[1])
                # if box_size > box_threshold:
                #     box_tracker[box_id] += 1
                if not face_book.contains(box_id):
                    for b in buffer:
                        if not any(existing_box[4] == box_id for existing_box in b["boxes"]):
                            b["boxes"].append(box[:5])
            buffer.append({"boxes": [box[:5] for box in result_boxes], "orig_img": result.orig_img})
            result = next(results_generator, None)
            working = True
            while len(buffer) > buffer_size or (result is None and len(buffer) > 0):
                current_image = buffer.pop(0)
                boxes = current_image["boxes"]
                frame = current_image["orig_img"]
                blur = cv2.resize(frame, (0, 0), fx=1.0 / pixel_size, fy=1.0 / pixel_size)
                blur = cv2.resize(blur, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask[:, :] = 0  # Set all values to 0
                face_book.cleanup()
                for box in boxes:
                    box_id = int(box[4])
                    box_size = (box[2] - box[0]) * (box[3] - box[1])
                    
                    # print("Box tracker:", box_tracker[box_id], "box id:" , box_id)
                    # if box_size > 10000 and box_tracker[box_id] < fps * 0.5:
                    #     print("Skipping box with id", box_id, "and size", box_size, "and count", box_tracker[box_id])
                    #     continue
                    if box_size > box_threshold:
                        x1, y1, x2, y2 = self.enlarge_box(box, frame_width, frame_height)
                        cropped_box = frame[y1:y2, x1:x2]
                        
                        face_result = self.model2(cropped_box)
                        print("Face result:", face_result)
                        print("Face result boxes:", face_result[0].boxes.data.cpu().numpy()[:5])
                        face_boxes = face_result[0].boxes.data.cpu().numpy()
                        if face_result[0].boxes:
                            print("Face detected in the box.")
                            x1, y1, x2, y2 = self.enlarge_box(face_boxes[0], frame_width, frame_height)
                            face_book.register(*map(int, [x1, y1, x2, y2]), hp=max(1, int(fps)), delta=frame_height / 12, fps=fps, id=box_id)
                            cropped_box - cv2.rectangle(cropped_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            plt.imshow(cropped_box)
                            plt.show()

                    else:
                        face_book.register(*map(int, box[:4]), hp=max(1, int(fps)), delta=frame_height / 12, fps=fps, id=box_id)
                for face in face_book:
                    x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), 255 * face.alpha, -1)
                    if face.debug:
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), face.color, 2)
                mask = cv2.GaussianBlur(mask, (mask_blur_ksize, mask_blur_ksize), 0)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = mask / 255.0
                frame = (frame * (1 - mask) + blur * mask).astype(np.uint8)
                output_video.write(frame)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    working = False
                    print("Window closed")
                    break
            if not working:
                break

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        return output_path

face_mozaic = FaceMozaic()
output_video_path = face_mozaic.add_mozaic(sys.argv[1])
print("Video successfully saved to:", output_video_path)
