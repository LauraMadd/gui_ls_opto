import serial
import time
import simpleaudio as sa


class Lights_control():
    
    def __init__(self, port='COM26'):
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
        self.arduino.write(str.encode('1'))
        print("Laser @ 488 nm On ")
        
    def green(self):
        self.arduino.write(str.encode('2'))
    
    def dark(self):
        self.arduino.write(str.encode('0'))
    
  
    def brightfield_off(self):
        self.arduino.write(str.encode('3'))
        
    def brightfield_on(self):
        self.arduino.write(str.encode('7'))
        
    def brightfield_on_25(self):
        self.arduino.write(str.encode('4'))
        
    def brightfield_on_50(self):
        self.arduino.write(str.encode('5'))
        
    def brightfield_on_75(self):
        self.arduino.write(str.encode('6'))
        
    def close(self):
        self.arduino.close()