import serial
import time


class Filters():
    """Class to control filters slider
    """

    def __init__(self, port='COM5'):
        self.slider = serial.Serial(port)
        self.home = str.encode('0ho0')
        self.slider.write(self.home)
        self.position_0 = str.encode('0ma00000000')
        self.position_1 = str.encode('0ma00000020')
        self.position_2 = str.encode('0ma00000040')
        self.position_3 = str.encode('0ma00000060')


    def move(self, position):
        if position == 0 : self.slider.write(self.position_0)
        elif position == 1 : self.slider.write(self.position_1)
        elif position == 2 : self.slider.write(self.position_2)
        elif position == 3 : self.slider.write(self.position_3)
        else: print('\n Wrong input for filters slider. \n')

    def filter_520(self):
        self.slider.write(self.position_1)

    def filter_IRcutoff(self):
        self.slider.write(self.position_1)


    def filter_620(self):
        self.slider.write(self.position_2)

    def filter_dark(self):
        self.slider.write(self.position_3)

    def filter_reset(self):
         self.slider.write(self.home)
         print("Reset filter")

    def test(self):
        self.slider.write(self.position_0)
        time.sleep(2)
        self.slider.write(self.position_1)
        time.sleep(2)
        self.slider.write(self.position_2)
        time.sleep(2)
        self.slider.write(self.position_3)
        time.sleep(2)
        self.slider.write(self.position_0)
        time.sleep(2)


    def close(self):
        self.slider.close()
