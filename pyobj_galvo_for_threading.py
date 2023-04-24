# -*- coding: utf-8 -*-
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
# import lights_control_arduino as lights
# reload(lights)
# import filters
# reload(filters)




class xy_scan(qtc.QObject):
    """ Planar scan + trigger.
     This class uses exactly the same functions in galvos.
     The reason to create it is to inherit from QObject the easy functions,
    that help putting it in a background thread.
    """

    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1', analog_out_channel=['ao0','ao3'],analog_in_channel=['ai1'],task_name=[ 'analog_task_scan_xy', 'analog_read'], parent=None):

        super(self.__class__, self).__init__(parent)
         # changed class for shutter 
        self.galvo = galvos.multi_DAQ_analogOut_shutter_rec(device_name, analog_out_channel,analog_in_channel,tasks_names=task_name)
        self.frequency_im = 8.49
        self.V_max = 1.4
        self.V_min = -1.4
        self.planes = 10
        self.t0_im=[]

    @qtc.pyqtSlot()
    def scan(self):
        time.sleep(2)
        self.galvo.single_plane_acquisitions(self.frequency_im,self.V_max,self.V_min)
        
    @qtc.pyqtSlot()
    def scan_numbered(self):
        # self.t0_im=[]
        time.sleep(2)
        self.t0_im,self.shutter_rec=self.galvo.single_plane_acquisitions_numbered(self.frequency_im,\
                                                    self.V_max,\
                                                    self.V_min,\
                                                    self.planes)
        # t0_im =str(datetime.time(datetime.now()))
        # self.t0_im.append(t0_im)
        print('start imaging',self.t0_im)
                                              
                                              
    #new functions for shutter                                          
                                              
    @qtc.pyqtSlot()
    def scan_numbered_single_pulse(self):
        time.sleep(2)
        self.galvo.single_plane_acquisitions_numbered_with_single_illumination(\
                                                                 self.frequency_im,\
                                                                 self.t_on_2p,\
                                                                 self.V_max,\
                                                                 self.V_min,\
                                                                 self.planes)
                                                                                                        
  #new functions for shutter                                               
    @qtc.pyqtSlot()
    def scan_numbered_train_pulse(self):
        time.sleep(2)
        self.galvo.single_plane_acquisitions_numbered_with_train_illumination(\
                                                                 self.frequency_im,\
                                                                 self.t_on_2p,\
                                                                 self.V_max,\
                                                                 self.V_min,\
                                                                 self.planes,\
                                                                 self.n_pulses)                                            
                                              
    @qtc.pyqtSlot()
    def clear_timing(self):
        # time.sleep(2)
        self.t0_im=[]
                                              
                                                                                                                                      
                                              
    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()
    
    
        
        

class single_scan_etl(qtc.QObject):
    """ Single scan navigation for ETL 
    Same explanation as before.
    """

    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1',
                        channels = 'ao1',
                        task_name = 'etl',
                        parent=None):

        super(self.__class__, self).__init__(parent)

        self.galvo = galvos.DAQ_analogOut(device_name, channels, task_name)
        self.value = 0

    @qtc.pyqtSlot()
    def constant(self):
        self.galvo.constant(self.value)
    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()
        
        
class single_scan_galvo_z(qtc.QObject):
    """ Single scan navigation for galvo ao2
    Same explanation as before.
    """

    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1',
                        channels = 'ao2',
                        task_name = 'z',
                        parent=None):

        super(self.__class__, self).__init__(parent)

        self.galvo = galvos.DAQ_analogOut(device_name, channels, task_name)
        self.value = 0

    @qtc.pyqtSlot()
    def constant(self):
        self.galvo.constant(self.value)
    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()        
        
        
        
        

        
        
        

class volume_scan(qtc.QObject):
    """ This class acquires volumes 
    """
    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1',
                        channels = ['ao0', 'ao1', 'ao2', 'ao3'],
                        task_name = 'xyz_scan',
                        parent=None):

        super(self.__class__, self).__init__(parent)

        self.galvo = galvos.multi_DAQ_analogOut(device_name, channels)

        self.frequency = 1.
        self.planes = 10
        self.V_plane_min = -1
        self.V_plane_max = 1
        self.V_z_min = -1
        self.V_z_max = 1
        self.V_etl_min = 0
        self.V_etl_max = 4

        self.V_etl_slope = .475
        self.V_etl_offset = 1.306
        
        self.volumes = 10
        self.t0_zstack=[]

    @qtc.pyqtSlot()
    def acquire_volume(self):
        time.sleep(2)
        self.t0_zstack=self.galvo.single_volume(self.frequency, self.planes, \
                                    self.V_plane_min, self.V_plane_max, \
                                    self.V_z_min, self.V_z_max,\
                                    self.V_etl_min, self.V_etl_max,
                                    self.V_elt_slope, self.V_etl_offset)

    @qtc.pyqtSlot()
    def acquire_volumes(self):
        time.sleep(2)
        self.galvo.acquisition_signals(self.frequency, self.planes, \
                                   self.V_plane_min, self.V_plane_max, \
                                   self.V_z_min, self.V_z_max,\
                                   self.V_etl_min, self.V_etl_max,
                                   self.V_etl_slope, self.V_etl_offset)
    @qtc.pyqtSlot()
    def acquire_volumes_numbered(self):
        time.sleep(2)
        self.t0_zstack=self.galvo.acquisition_signals_numbered(self.frequency, self.planes, \
                                   self.V_plane_min, self.V_plane_max, \
                                   self.V_z_min, self.V_z_max,\
                                   self.V_etl_min, self.V_etl_max,
                                   self.V_etl_slope, self.V_etl_offset, 
                                   self.volumes)

    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()

    @qtc.pyqtSlot()
    def clear_timing_zstack(self):
        # time.sleep(2)
        self.t0_zstack=[]
#--------------------------------------------------------

class volume_scan_multicolor(qtc.QObject):
    """ This class acquires volumes 
    """
    signalStatus = qtc.pyqtSignal(str)

    def __init__(self,lights,filters, device_name='Dev1',
                        channels = ['ao0', 'ao1', 'ao2', 'ao3'],
                        task_name = 'xyz_scan_multicolor',
                        parent=None):

        super(self.__class__, self).__init__(parent)

        self.galvo = galvos.multi_DAQ_analogOut(device_name, channels)

        self.frequency = 1.
        self.planes = 10
        self.V_plane_min = -1
        self.V_plane_max = 1
        self.V_z_min = -1
        self.V_z_max = 1
        self.V_etl_min = 0
        self.V_etl_max = 4

        self.V_etl_slope = .475
        self.V_etl_offset = 1.306
        
        self.volumes = 10
        self.t0_zstack=[]
        self.delta_t=30
        self.n_volumes=10

        #Initialization lasers + filters 
    
        self.filters=filters
        self.lights=lights

        
        

    @qtc.pyqtSlot()
    def acquire_volume(self):
        time.sleep(2)
        self.t0_zstack=self.galvo.single_volume(self.frequency, self.planes, \
                                    self.V_plane_min, self.V_plane_max, \
                                    self.V_z_min, self.V_z_max,\
                                    self.V_etl_min, self.V_etl_max,
                                    self.V_elt_slope, self.V_etl_offset)


    @qtc.pyqtSlot()
    def acquire_volumes_multicolor(self):

        
        for i in range (self.n_volumes): 
            print("entrato")
        # self.filters.filter_620()
        # self.lights.green()
        # self.acquire_volume()
        # self.lights.dark()
        # self.stop()
        # print('Blue channel acquisition')
        # self.filters.filter_520()
        # self.lights.blue()
        # #add time stamp here
        # self.acquire_volume()
        # self.lights.dark()
        # self.stop()
            time.sleep(self.delta_t)




    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()

    @qtc.pyqtSlot()
    def clear_timing_zstack(self):
        # time.sleep(2)
        self.t0_zstack=[]
#----------------------------------------------------------------
class volume_scan_shutter(qtc.QObject):
    """ This class acquires volumes and records the shutter activity 
    """
    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1',
                        analog_out_channel=['ao0','ao1','ao2','ao3'],analog_in_channel=['ai1'], tasks_names=['analog_xyz_scan','analog_read_xyz_scan'],parent=None):

        super(self.__class__, self).__init__(parent)

        self.galvo = galvos.multi_DAQ_volume_shutter_rec(device_name, analog_out_channel,analog_in_channel,tasks_names=tasks_names)

        self.frequency = 1.
        self.planes = 10
        self.V_plane_min = -1
        self.V_plane_max = 1
        self.V_z_min = -1
        self.V_z_max = 1
        self.V_etl_min = 0
        self.V_etl_max = 4

        self.V_etl_slope = .475
        self.V_etl_offset = 1.306
        
        self.volumes = 10
        self.t0_zstack=[]



    @qtc.pyqtSlot()
    def acquire_volumes_numbered(self):
        time.sleep(2)
        self.t0_zstack,self.shutter_rec=self.galvo.acquisition_signals_numbered(self.frequency, self.planes, \
                                   self.V_plane_min, self.V_plane_max, \
                                   self.V_z_min, self.V_z_max,\
                                   self.V_etl_min, self.V_etl_max,
                                   self.V_etl_slope, self.V_etl_offset, 
                                   self.volumes)

    @qtc.pyqtSlot()
    def stop(self):
        self.galvo.stop()

    @qtc.pyqtSlot()
    def close(self):
        self.galvo.close()

    @qtc.pyqtSlot()
    def clear_timing_zstack(self):
        # time.sleep(2)
        self.t0_zstack=[]

class shutter(qtc.QObject):
    """ This class controls the shutter opening and closing through a digital output channel of the DAQ
    """

    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, device_name='Dev1', digital_channels=['do0'],task_name = 'shutter', parent=None):

        super(self.__class__, self).__init__(parent)
         # changed class for shutter 
        self.tasks_names=[task_name]
        self.shutter = galvos.single_DAQ_shutter(device_name,digital_channels, self.tasks_names)
        self.t_tot_single=10
        self.t_tot=1
        self.t_on=0.001
        self.n_pulses=10
        self.frequency=1.
        self.t_2P_single = []
        self.t_2P_train = []

    @qtc.pyqtSlot()
    def single_pulse(self):
        # time.sleep(2)
        
        t_2P_single=self.shutter.single_illumination(self.t_tot_single, self.t_on)
       
        self.t_2P_single.append(t_2P_single)
        print('start 2P single',self.t_2P_single)
        
    @qtc.pyqtSlot()
    def train_illumination(self):
        # time.sleep(2)
        # self.t_2P_train = []
        self.t_2P_train=self.shutter.train_illumination( self.frequency, self.t_on, self.n_pulses)
        # t_2P_train =str(datetime.time( datetime.now()))
        # self.t_2P_train.append(t_2P_train)
        print('start 2P train',self.t_2P_train)
        
        
    @qtc.pyqtSlot()
    def clear_timing(self):
        # time.sleep(2)
        self.t_2P_single = []
        self.t_2P_train = []
        
    @qtc.pyqtSlot()
    def close(self):
        # time.sleep(2)
        self.shutter.close()

        
        
           

        
        
