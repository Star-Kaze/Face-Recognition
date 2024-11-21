import cv2
import face_recognition
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel,QPushButton
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import sys

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.faces_exist = False
        self.image_list = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Face Recognition")
        self.setGeometry(100, 100, 1100, 600)
        self.setStyleSheet("""
        QMainWindow {
            background-color: lightpink;
        }
        QPushButton {
            background-color: lightpink;
            font-size: 20px;
            color: purple;
            border: 2px solid purple;
            border-radius: 10px;
            padding: 10px;
            margin: 10px;
        }
        QPushButton:hover {
            background-color: purple;
            color: lightpink;
        }
        QPushButton:pressed {
            background-color: darkmagenta;
            color: white;
        }
        """)

        take_pic_but = QPushButton("Take Picture", self)
        take_pic_but.setGeometry(800, 260, 200, 80)
        take_pic_but.clicked.connect(self.take_photo)

        self.cam_label = QLabel(self)
        self.cam_label.setGeometry(60, 60, 640, 480)
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        timer = QTimer(self)
        timer.timeout.connect(self.show_webcam)
        timer.start(10)

    def show_webcam(self):
        ret, frame = self.cam.read()
        # print(frame.shape)
        if ret:
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                largest_face_area = 0
                largest_face = None
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_area = w * h

                        # Update largest face if this has a bigger area
                        if face_area > largest_face_area:
                            largest_face_area = face_area
                            largest_face = (x, y, w, h)

                    # If a largest face is found, draw rectangle and potentially capture it
                    if largest_face:
                        x, y, w, h = largest_face
                        # add images to image_list
                        self.image_list["cut_face"] = frame[y:y + h,x:x + w]
                        self.image_list["full_image"] = frame
                        self.faces_exist = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    self.faces_exist = False

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format.Format_RGB888)
                qpix = QPixmap.fromImage(qimg)
                self.cam_label.setPixmap(qpix)

    def take_photo(self):
        if self.faces_exist:
            cut_face_rgb = cv2.cvtColor(self.image_list["cut_face"], cv2.COLOR_BGR2RGB)
            full_image_rgb = cv2.cvtColor(self.image_list["full_image"], cv2.COLOR_BGR2RGB)
            cv2.imwrite('./.cut_face.jpg', cut_face_rgb)
            cv2.imwrite('./.full_image.jpg', full_image_rgb)
            print("Photos saved successfully")
            print("Cut face color format:", cut_face_rgb.shape)
            print("Full image color format:", full_image_rgb.shape)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())