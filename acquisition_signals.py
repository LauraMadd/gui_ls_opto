import importlib
import numpy as np
import matplotlib.pyplot as plt
from ETL import Opto as etl
import galvo_control as galvo
import time
from importlib import reload
reload(galvo)

# ------------------------------------------------------------------------------

# Procedure to take an image with SOLIS and triggers from galvo:
# 1- prepare SOLIS parameters
# 2- launch galvos with those parameters
# 
scan_plane = galvo.multi_DAQ_analogOut(channels = ['ao0', 'ao3'])
scan_z = galvo.DAQ_analogOut(channel_name = 'ao2')
defocus_lens = etl(port = 'COM24')
defocus_lens.connect()


def volume_acquisition_signals(scan_plane, scan_z, defocus_lens, \
                            V_z_min, V_z_max, step_z, \
                            frequency = 1., repetitions=1, V_max=1, V_min=-1):
                                
    z_range = np.arange(V_z_min, V_z_max, step_z)    
    
    for z in (z_range):
        scan_z.constant(z)
        defocus_lens.current(z*76)
        time.sleep(0.005)
        scan_plane.single_plane_acquisitions(frequency, repetitions, \
                                            V_max, V_min)                              
    
    return None

def acquire_volumes(volumes_number, \
                    scan_plane, scan_z, defocus_lens, \
                    V_z_min, V_z_max, step_z, \
                    frequency = 1., repetitions=1, V_max=1, V_min=-1):
    start = time.time()                
    for i in range(volumes_number):
        volume_acquisition_signals(scan_plane, scan_z, defocus_lens, \
                            V_z_min, V_z_max, step_z, \
                            frequency, repetitions, V_max, V_min)
        print(i)
    print(time.time()-start)
    return None
    
def planes_at_variable_intervals(galvo_plane, V_min, V_max, image_frequency, \
                                    interval, num_planes,  repetitions = 1):
    for i in range(num_planes):
        galvo_plane.single_plane_acquisitions(image_frequency, repetitions, \
                                            V_max, V_min)
        galvo_plane.single_value('ao0', V_min-1.5)
        time.sleep(interval)
    return None

def volumes_at_variable_intervals(scan_plane, scan_z, defocus_lens, \
                            V_z_min, V_z_max, step_z, \
                            frequency, V_max, V_min,\
                            num_volumes, interval, repetitions = 1):
    start = time.time()
    
    for i in range(num_volumes):
        volume_acquisition_signals(scan_plane, scan_z, defocus_lens, \
                            V_z_min, V_z_max, step_z, \
                            frequency, repetitions, V_max, V_min)
        scan_plane.single_value('ao0', V_min-1.5)
        time.sleep(interval)
        
    print(time.time()-start)
    
    return None

def volume_acquisition_without_etl(scan_plane, scan_z, \
                            V_z_min, V_z_max, step_z, \
                            frequency = 1., repetitions=1, V_max=1, V_min=-1):
    
    z_range = np.arange(V_z_min, V_z_max, step_z)    
    
    for z in (z_range):
        scan_z.constant(z)
        time.sleep(.3)
        scan_plane.single_plane_acquisitions(frequency, repetitions, \
                                            V_max, V_min)
                                                
    return None


#acquire_volumes(1, scan_plane, scan_z, defocus_lens, V_z_min = -0.7, V_z_max = 0.7, step_z = 0.0125, frequency = 4.642526, V_max = 1.32, V_min = -1.27)
























