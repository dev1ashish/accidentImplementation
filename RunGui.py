import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import time
import zmq
from numpy.core.multiarray import ndarray
from pygame import mixer

from System.Data.CONSTANTS import *
from System.Controller.JsonEncoder import JsonEncoder

class WorkerThread(QObject):
    receive = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://"+GUIIP+":"+str(GUIPORT))

    @pyqtSlot()
    def run(self):
        while True:
            message = self.socket.recv_pyobj()  # receive a message json
            self.socket.send_pyobj("")
            self.receive.emit(message)


class VideoDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Display')
        
        # Video display area
        self.results = QListWidget(self)
        self.results.resize(800, 600)
        self.results.move(0, 0)
        self.results.itemClicked.connect(self.listwidgetClicked)

    def playVideo(self, video):
        for i in range(len(video)):
            cv2.imshow('Frame', video[i])
            if cv2.waitKey(31) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def decode(self, msg):
        func = msg[FUNCTION]
        if func == REP_VIDEO:
            self.playVideo(msg[FRAMES])
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    form = VideoDisplay()
    form.show()
    sys.exit(app.exec_())