import cv2

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces_in_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.detect_faces(frame)
            self.mosaic_faces(frame, faces)

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        return faces


    def mosaic_faces(self, image, faces, mosaic_size=(16, 16)):
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]

            small_face = cv2.resize(face, mosaic_size, interpolation=cv2.INTER_NEAREST)
            mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)

            image[y:y+h, x:x+w] = mosaic_face

face_detector = FaceDetector()
video_path = 'test2.mp4'
face_detector.detect_faces_in_video(video_path)
