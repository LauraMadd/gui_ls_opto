

from importlib import reload
#new for shuttter
import galvos_with_shutter as galvos
reload(galvos)
from PyQt5 import QtCore as qtc
import time
import numpy as np 
from datetime import datetime
import time
import zmq
import pickle
import json



class shutter_server(qtc.QObject):
    """ Planar scan + trigger.
     This class uses exactly the same functions in galvos.
     The reason to create it is to inherit from QObject the easy functions,
    that help putting it in a background thread.
    """

    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1', analog_channels=['ai1'],task_name=['analog_read'], parent=None):

        super(self.__class__, self).__init__(parent)
         # changed class for shutter 
        self.galvo = galvos.Read_Shutter(device_name, analog_channels,tasks_names=task_name)
      
      
     
        #new functions for shutter   
  
        # self.values_read=[]
        # self.values_read=np.zeros(int(10**3), dtype=np.float64)        
        
        
    @qtc.pyqtSlot()
    def read_shutter_finite(self):
        self.values_read=self.galvo.rec_shutter_finite()
        
        return self.values_read
            
    @qtc.pyqtSlot()
    def read_shutter_continuous(self):
        
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")        
                
        while True:
            message = socket.recv()
            print("Received request: %s" % message)
            self.values_read=self.galvo.rec_shutter_continuous_bis()
        #  Do some 'work'
        #    time.sleep(1)
    
        #  Send reply back to client
            socket.send(self.values_read.dumps())
            print('Shutter read')
        #     time.sleep(1.0)#
        # self.values_read=self.galvo.rec_shutter_continuous()
        # self.values_read=self.galvo.rec_shutter_continuous_bis()
        # return self.values_read
       
        # print('shape rec in pyobj', self.values_read.shape)
        # return self.values_read

        
    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()   
        
    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close() 