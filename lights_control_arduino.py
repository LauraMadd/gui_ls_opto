import serial
import time
# import simpleaudio as sa


class Lights_control():
    """Class to control light sources= 488 nm, 561 nm and white light trough arduino
        board. This code relies on the arduino code to programm the board.
    """

    def __init__(self, port='COM6'):
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

    def blue(self):
        """Blue laser on
        """
        self.arduino.write(str.encode('1'))
        print("Laser @ 488 nm On ")

    def green(self):
        """Green laser on
        """
        self.arduino.write(str.encode('2'))
        print("Laser @ 561 nm On ")

    
    def blue_green(self):
        """Blue and green laser on
        """
        self.arduino.write(str.encode('8'))
       
        print("Laser @ 488 and 561 nm On ")

    def dark(self):
        """Both lasers off
        """
        self.arduino.write(str.encode('0'))


    def brightfield_0(self):
        """White light off
        """
        self.arduino.write(str.encode('3'))

    def brightfield_on_100(self):
        """White light on 100%
        """
        self.arduino.write(str.encode('7'))

    def brightfield_on_25(self):
        """White light on 25%
        """
        self.arduino.write(str.encode('4'))

    def brightfield_on_50(self):
        """White light on 50%
        """
        self.arduino.write(str.encode('5'))

    def brightfield_on_75(self):
        """White light on 75%
        """
        self.arduino.write(str.encode('6'))

    def pump_on(self):
        """Activates pump for the flow chamber
            """
        self.arduino.write(str.encode('e'))

    def pump_off(self):
        """Shuts off pump for the flow chamber
        """
        self.arduino.write(str.encode('f'))
    

    def close(self):
        """Close communication with arduino
        """
        self.arduino.close()
