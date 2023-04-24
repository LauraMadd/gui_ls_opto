import serial
import time
# import simpleaudio as sa


class Matrix_control():
    """Class to control light sources= 488 nm, 561 nm and white light trough arduino
        board. This code relies on the arduino code to programm the board.
    """

    def __init__(self, port='COM8'):
        self.arduino = serial.Serial(port)
    def test(self):
        for i in range(10):
            self.arduino.write(str.encode('1'))
            self.play_obj = self.wave_obj.play()
            self.play_obj.wait_done()  # Wait until sound has finished playing
            time.sleep(1)
            self.arduino.write(str.encode('0'))
            self.play_obj = self.wave_obj.play()
            self.play_obj.wait_done()  # Wait until sound has finished playing
            time.sleep(1)




    def bf_off(self):
        """Red lights off
        """
        self.arduino.write(str.encode('d'))

    def bf_all_on(self):
        """Red lights on
        """
        self.arduino.write(str.encode('e'))
        
    def coordinate_shine(self,x,y):
        """shining a single led
        """
        self.arduino.write(str.encode('h'+str(y*16+x))) 
    def close(self):
        """Close communication with arduino
        """
        self.arduino.close()
