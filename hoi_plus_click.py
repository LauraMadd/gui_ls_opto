from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import numpy as np


class HandsOn(qtw.QLabel):
    """
    Class to manipulate images:
    - drag to select a ROI
    - Click to register coordinates
    """
    def __init__(self, picture):
        """
        - picture is a numpy 2D array, it's assume to be a 16bit image
        """
        super(HandsOn, self).__init__()
        # renormalize picture in 16 bit to better see it
        self.array = ((
            (picture[:, :] - np.amin(picture))
            / (np.amax(picture) - np.amin(picture)))
            * (2**16-1)).astype(np.uint16)

        self.picture = qtg.QImage(
                                self.array, self.array.shape[1],
                                self.array.shape[0],
                                2 * self.array.shape[1],
                                qtg.QImage.Format_Grayscale16)
        self.pix = qtg.QPixmap(self.picture)
        self.setPixmap(self.pix)
        # xStart and yStart are in row-cols notation, numpy-friendly
        self.xStart, self.yStart = 0., 0.
        self.xEnd, self.yEnd = 0., 0.
        self.rubberBand = qtw.QRubberBand(qtw.QRubberBand.Rectangle, self)
        self.origin = qtc.QPoint()
        self.drag_flag = True
        self.coords = []
        self.stack_index = 0
        self.stack_index_in_microns = 0

    def mousePressEvent(self, event):
        if event.button() == qtc.Qt.LeftButton:
            if self.drag_flag is True:
                self.xStart, self.yStart = event.pos().y(), event.pos().x()
                #LM cartesian 
                # self.xStart, self.yStart = event.pos().x(), event.pos().y()
                self.origin = qtc.QPoint(event.pos())
                self.rubberBand.setGeometry(
                                        qtc.QRect(self.origin, qtc.QSize()))
                self.rubberBand.show()
            else:
                self.coords.append([event.pos().y(), event.pos().x(), self.stack_index])
                self.draw_circle(event.pos().x(), event.pos().y())
                #LM cartesian
                # self.coords.append([event.pos().x(), event.pos().y(), self.stack_index])
                print(self.coords)

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            if self.drag_flag is True:
                self.rubberBand.setGeometry(
                                         qtc.QRect(
                                                    self.origin,
                                                    event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == qtc.Qt.LeftButton:
            if self.drag_flag is True:
                self.xEnd, self.yEnd = event.pos().y(), event.pos().x()
            temp_x, temp_y = 0., 0.
            if self.xStart > self.xEnd:
                temp_x = self.xStart
                self.xStart = self.xEnd
                self.xEnd = temp_x
            if self.yStart > self.yEnd:
                temp_y = self.yStart
                self.yStart = self.yEnd
                self.yEnd = temp_y

    def draw_circle(self, a, b, r=5):
        """ Create a painter that draws a circle or radii a and b.
        Then delete the painter, so that another one can be created at any moment
        and we don't have too many actors lying around.
        """
        painter = qtg.QPainter(self.pix)
        painter.setPen(qtg.QPen(qtg.QColor(0, 166, 217, 150),  2, qtc.Qt.SolidLine))
        painter.setBrush(qtg.QBrush(qtg.QColor(0, 166, 217, 150), qtc.Qt.Dense4Pattern))
        painter.drawEllipse(qtc.QPoint(a, b), r, r)
        self.setPixmap(self.pix)
        del painter

    def update_image(self, picture):
        """
        To display a new image:
        1 - rescale the np array
        2 - put the matrix in an image
        3 - put the image in a pixmap
        """
        self.array = (
                ((picture[:, :] - np.amin(picture))
                    / (np.amax(picture) - np.amin(picture)))
                    * (2**16-1)).astype(np.uint16)
        self.picture = qtg.QImage(
                                    self.array, self.array.shape[1],
                                    self.array.shape[0],
                                    2 * self.array.shape[1],
                                    qtg.QImage.Format_Grayscale16)
        self.pix = qtg.QPixmap(self.picture)
        self.setPixmap(self.pix)
        for i, coord in enumerate(self.coords):
            if (self.coords[i][2] == self.stack_index):
    
                self.draw_circle(self.coords[i][1], self.coords[i][0])

    def empty_coords_list(self):
        print('empty')
        self.coords = []
        self.pix = qtg.QPixmap(self.picture)
        self.setPixmap(self.pix)

    def forget_last_point(self):
        if self.coords != []:
            del self.coords[-1]
            self.pix = qtg.QPixmap(self.picture)
            self.setPixmap(self.pix)
            for i, coord in enumerate(self.coords):
                if (self.coords[i][2] == self.stack_index):
                    self.draw_circle(self.coords[i][1], self.coords[i][0])


class SLM_prewview(qtw.QLabel):
    """
    Class to manipulate images:
    - drag to select a ROI
    - Click to register coordinates
    """
    def __init__(self, picture):
        """
        - picture is a numpy 2D array, it's assume to be a 16bit image
        """
        super(HandsOn, self).__init__()
        # renormalize picture in 16 bit to better see it
        self.array = ((
            (picture[:, :] - np.amin(picture))
            / (np.amax(picture) - np.amin(picture)))
            * (2**16-1)).astype(np.uint16)

        self.picture = qtg.QImage(
                                self.array, self.array.shape[1],
                                self.array.shape[0],
                                2 * self.array.shape[1],
                                qtg.QImage.Format_Grayscale16)
        self.pix = qtg.QPixmap(self.picture)
        self.setPixmap(self.pix)
        # xStart and yStart are in row-cols notation, numpy-friendly
        self.xStart, self.yStart = 0., 0.
        self.xEnd, self.yEnd = 0., 0.
        self.rubberBand = qtw.QRubberBand(qtw.QRubberBand.Rectangle, self)
        self.origin = qtc.QPoint()
        self.drag_flag = True
        self.coords = []
        self.stack_index = 0
        
    def update_image(self, picture):
        self.array = (
                ((picture[:, :] - np.amin(picture))
                    / (np.amax(picture) - np.amin(picture)))
                    * (2**16-1)).astype(np.uint16)
        self.picture = qtg.QImage(
                                    self.array, self.array.shape[1],
                                    self.array.shape[0],
                                    2 * self.array.shape[1],
                                    qtg.QImage.Format_Grayscale16)
        self.pix = qtg.QPixmap(self.picture)
        self.setPixmap(self.pix)