import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import face_recognition
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel,QPushButton,QStackedLayout,QWidget
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

        central_widget = QStackedLayout()
        self.setLayout(central_widget)

        self.main_window = QWidget()
        self.main_window_setup()
        central_widget.addWidget(self.main_window)


    def main_window_setup(self):
        take_pic_but = QPushButton("Take Picture", self.main_window)
        take_pic_but.setGeometry(800, 260, 200, 80)
        take_pic_but.clicked.connect(self.take_photo)

        self.cam_label = QLabel(self.main_window)
        self.cam_label.setGeometry(60, 60, 640, 480)
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0

        timer = QTimer(self.main_window)
        timer.timeout.connect(self.show_webcam)
        timer.start(10)

    def show_webcam(self):
        ret, frame = self.cam.read()
        # print(frame.shape)
        if ret:
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                cam_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(cam_image, model="hog")
                largest_face_area = 0
                largest_face = None
                if face_locations:
                    for (top, right, bottom, left) in face_locations:
                        face_area = (right - left) * (bottom - top)

                    # Update largest face if this has a bigger area
                    if face_area > largest_face_area:
                        largest_face_area = face_area
                        largest_face = (top, right, bottom, left)

                    # If a largest face is found, draw rectangle and potentially capture it
                    if largest_face:
                        top, right, bottom, left = largest_face
                        cv2.rectangle(cam_image, (left, top), (right, bottom), (0, 255, 0), 2)
                        # add images to image_list
                        self.image_list["cut_face"] = cam_image[top:bottom,left:right]
                        self.image_list["full_image"] = cam_image
                        self.faces_exist = True
                else:
                    self.faces_exist = False

                qimg = QImage(cam_image.data, cam_image.shape[1], cam_image.shape[0], QImage.Format.Format_RGB888)
                qpix = QPixmap.fromImage(qimg)
                self.cam_label.setPixmap(qpix)

    def take_photo(self):
        if self.faces_exist:
            cut_face_rgb = cv2.cvtColor(self.image_list["cut_face"], cv2.COLOR_BGR2RGB)
            full_image_rgb = cv2.cvtColor(self.image_list["full_image"], cv2.COLOR_BGR2RGB)
            cv2.imwrite('./.cut_face.jpg', cut_face_rgb)
            cv2.imwrite('./.full_image.jpg', full_image_rgb)
            print("Photos saved successfully")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())