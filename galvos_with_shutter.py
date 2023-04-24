import numpy as np
import nidaqmx
import nidaqmx.stream_writers
import nidaqmx.stream_readers
from scipy.signal import sawtooth, square, triang
import matplotlib.pyplot as plt
from datetime import datetime


#-------------------------------------------------------------------------------


class DAQ_analogOut(object):
    """ Class to control the DAQ
    """
    def __init__(self, device_name='Dev1',
                       channel_name=['ao0'],
                    task_name = 'name'):

        """ Constructor:
                - device_name = obvious;
                - channel_name = the channels that are hard wired to externad hardware;
                - task_name = just a name to allow multiple tasks to be creaed
        """
        self.device_name = device_name
        self.channel_name = channel_name
        self.task = nidaqmx.Task(task_name)
        self.task.ao_channels.add_ao_voltage_chan(device_name+'/'+channel_name)

    def close(self):
        """ Completely free the task.
        """
        self.task.stop()
        self.task.close()

    def constant(self, value):
        """ Output a constant value.
        """
        assert(value<10. or value >-10.), \
                                    '\n max output 10V: you put   %f' %value
        self.task.write(value)

    def sine(self, frequency = 1., V_max=5, V_min=-5,\
                    offset = 0, num_samples = 10**5):

        """ Generate sine wave:
                - frequency = of the signal, [Hz];
                - repetitions = how many times repeat the signal?
                - V_max = obvious;
                - V_min = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <=10.), '\n max output 10V: you put   %f' %V_max
        assert(V_max >=-10.), '\n max output 10V: you put %f' %V_min

        self.t = np.linspace(0, 1/frequency, int(num_samples))

        self.signal = (np.sin(2*np.pi*self.t*frequency)+1) \
                                        *(V_max-V_min+1)/2. \
                                        + V_min + offset

        self.task.timing.cfg_samp_clk_timing(rate = num_samples*frequency,\
                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS)

        writer = nidaqmx.stream_writers.AnalogSingleChannelWriter(\
                                        self.task.out_stream, auto_start=True)

        writer.write_many_sample(self.signal)


# ------------------------------------------------------------------------------

class multi_DAQ_analogOut(object):
    """ Class to control the light sheet acquisition. The DAQ controls plane galvo, 
    depth galvo, camera trigger, etl scan (analog channels). There are functions to acquire
    time lapses, volumes and time lapses of volumes. 
    """
    def __init__(self, device_name='Dev1', channels=['ao0','ao1','ao2','ao3']):

        """ Constructor:
                - device_name = obvious;
                - channels = list with channels' names...
                    -'ao0' for image scan
                    -'ao1' for depth scan
                    -'ao2' for etl scan
                    -'ao3' for camera trigger
            The logic is: open the DEVICE, select the OUTPUT CHANNELS that send data,
            define a TASK that contains the functions to transmit and manipulate signals.
        """
        self.device_name = device_name
        self.channels = channels
        self.task = nidaqmx.Task()
        for i in range (len(self.channels)):
            # add the active channels to the task
            self.task.ao_channels.add_ao_voltage_chan(\
                                device_name+'/'+self.channels[i])

        self.single_volume_done = False

    def stop(self):
        """ Stop the task.
            Close it (forget eerything).
            Then open again.
        """
        self.task.stop()
        self.task.close()
        self.task = nidaqmx.Task()
        for i in range (len(self.channels)):
            # add the active channels to the task
            self.task.ao_channels.add_ao_voltage_chan(\
                                self.device_name+'/'+self.channels[i])
            # default everything to 0
        if(len(self.channels) == 2):
            self.task.write([0, 0])
        else:
            self.task.write([0, 0, 0, 0])

    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.task.stop()
        self.task.close()

    def single_plane_acquisitions(self, frequency = 1.,\
                                  V_max = 1, V_min = -1):
        """ Acquire single planes (forever).
            - frequency = obvious
            - V_max = voltage when the light sheet is at the top
            - V_min = same, but at the bottom
        """
        assert(self.channels == ['ao0', 'ao3']), \
                "\n Select only channels \ao0\' & \'ao3\' for this modality"

        num_samples =  10**4 # default from the DAQ specifications

        self.t = np.linspace(0, 2/frequency, num_samples, dtype=np.float16)

        one_second = int(len(self.t)*frequency/2) # samples in one sec
        
        # triangular signal to scan the plane: every period 
        # (of the triang wave) scans 2 times (up and down)
        self.signal_1 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_1 = triang(int(num_samples))\
                     * (V_max - V_min)\
                     + V_min
        self.signal_1[:] = signal_1[:]

        # rect signal to define the trigger to the camera
        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t * frequency, duty = .1)+1)/2.
        signal_2[int(len(signal_2)*3/4+1):] = 0
        self.signal_2[:] = signal_2[:]

        # put them in a matrix to write with the task
        # (when write_many_samples is called)
        self.matrix = np.zeros((2, len(self.signal_1)))
        self.matrix[0,:] = self.signal_1
        self.matrix[1,:] = self.signal_2
        
        # CONTINUOUS means that will go on forever.
        # "num samples *2" because every channel transmits "samples", and we have 2 channels
        self.task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan=num_samples*2)

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)

        writer.write_many_sample(self.matrix)

    
    def single_plane_acquisitions_numbered(self, frequency = 1.,\
                                  V_max = 1, V_min = -1, num_images=10):
        """ Acquire a time lapse of one plane 
            - frequency = image acquisition frequency 
            - V_max = voltage when the light sheet is at the top
            - V_min = same, but at the bottom
            - num_images = images in time lapse 
        """
        assert(self.analog_channels == ['ao0', 'ao3']), \
                "\n Select only channels \ao0\' & \'ao3\' for this modality"

        num_samples =  10**4

        self.t = np.linspace(0, 2/frequency, num_samples, dtype=np.float16)

        one_second = int(len(self.t)*frequency/2) # samples in one sec
        
        
        self.signal_1 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_1 = triang(int(num_samples))\
                     * (V_max - V_min)\
                     + V_min
        self.signal_1[:] = signal_1[:]

        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t * frequency, duty = .1)+1)/2.
        signal_2[int(len(signal_2)*3/4+1):] = 0


        self.signal_2[:] = signal_2[:]

        self.matrix = np.zeros((2, len(self.signal_1)))
        self.matrix[0,:] = self.signal_1
        self.matrix[1,:] = self.signal_2
        
        
        # the difference with before is in samps_per_chan
        # by design,eery triang wave goes up and down, so scans 2 images
        # that's why there is a 2 there
        # also the sample_mode is FINITE now
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(num_samples*num_images/2))

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=True)
        
        self.t0_im=[]
        t0_im =str(datetime.time(datetime.now()))
        self.t0_im.append(t0_im)
        writer.write_many_sample(self.matrix)
        # print('\nScan done. \n', flush=True)
        print('\nScan done. \n')
        
        return self.t0_im
                                  

    def acquisition_signals(self, frequency = 1., planes = 10, \
                                   V_plane_min = -.5, V_plane_max = .5, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28):
        """ Acquire volumes (forever).
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
        """
        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 5.), \
                "\n Cannot send more than 5V to the ETL (channel 1)"
        assert(planes % 2 == 0), \
                "\n Cannot collect an odd number of planes"

        num_samples = 700*10**3 # int, from DAQ characteristics

        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)

        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        # it's a step function from Vmin to Vmax
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        self.signal_2 = np.zeros((len(steps_final)))*V_z_min
        for i in range(int(len(steps_final)/2)):
            if(steps_final[i]>.9):
                self.signal_2[i:] += z_step
        for i in range(int(len(steps_final)/2), len(steps_final)):
            if(steps_final[i]>.9):
                self.signal_2[i:] -= z_step

        # signal_3 is to change the ETL position
        # same as signal_2 but with different parameters
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step\

        self.signal_3[:] = (self.signal_2[:] + V_etl_offset) /V_etl_slope

        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2
        self.matrix[3,:] = self.signal_1

        # as beofore, define timings and samples length
        self.task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS)

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)

        writer.write_many_sample(self.matrix)
    
    def acquisition_signals_numbered(self, frequency = 1., planes = 10, \
                                   V_plane_min = -.5, V_plane_max = .5, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28,
                                   volumes=1):
        """ Acquire volumes (a certain number).
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
            - volumes = number of volumes to acquire
        """
        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 5.), \
                "\n Cannot send more than 5V to the ETL (channel 1)"
        assert(planes % 2 == 0), \
                "\n Cannot collect an odd number of planes"

        num_samples = 700*10**3 # it is considered an int -- check if enough!!!
        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)


        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        # self.signal_2 = np.zeros((len(steps_final)))*V_z_min
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] += z_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] -= z_step

        self.signal_2_b = np.ones((len(self.signal_0)))*V_z_min

        upflag = 1
        #inizia da mezzo duty
        for i in range(1, int((len(self.signal_1)/2))):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] += z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0
        upflag = 1
        for i in range(int(len(self.signal_1)/2), len(self.signal_1)):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] -= z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0





        # signal_3 is to change the ETL position
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step\

        self.signal_3[:] = (self.signal_2_b[:] + V_etl_offset) /V_etl_slope

        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2_b
        self.matrix[3,:] = self.signal_1

        # this is the only change from before.
        # FINITE acquisition, with different samps_per_chan
        # the logic of the "/2" is the same as for single planes:
        # the step functions of the ETL and z-galvo go up and down,
        # so a complete period covers 2 volumes
        self.task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(len(self.signal_0)*volumes/2))
                    # samps_per_chan=int(num_samples*volumes/2))
                    # samps_per_chan=volumes* int(len(self.t)/2))

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)

        self.t0_zstack=[]
        t0_zstack =str(datetime.time(datetime.now()))
        self.t0_zstack.append(t0_zstack)

        writer.write_many_sample(self.matrix)
        print ('etl_offset ', V_etl_offset, 'etl_slope',V_etl_slope, 'V_etl_min', V_etl_min)
        print('V_z_min', V_z_min, 'V_z_max', V_z_max, 'V_plane_max', V_plane_max, 'V_plane_min',V_plane_min, 'f',frequency, 'planes ',planes )

        return self.t0_zstack


    def single_volume(self, frequency = 1., planes = 10, \
                                   V_plane_min = -1, V_plane_max = 1, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28):
        """ Acquire a single volume.
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
        """
        self.single_volume_done = False


        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 4.99), \
                "\n Cannot send more than 5V to the ETL (channel 1)"

        num_samples = 700*10**3 # it is considered an int -- check if enough!!!
        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)


        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        self.signal_2 = np.ones((len(steps_final)))*V_z_min
        for i in range(int(len(steps_final)/2)):
            if(steps_final[i]>.9):
                self.signal_2[i:] += z_step
        for i in range(int(len(steps_final)/2), len(steps_final)):
            if(steps_final[i]>.9):
                self.signal_2[i:] -= z_step


        # signal 3  to ETL uses the fit parameters of the calibration ETL voltage 
        # galvo voltage.
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step

        self.signal_3[:] = (self.signal_2[:] + V_etl_offset) / V_etl_slope


        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2
        self.matrix[3,:] = self.signal_1

        # agai, the only change is in the sampls_per_chan
        self.task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                        samps_per_chan = planes * int(len(self.t)/2))
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)

        self.t0_zstack=[]
        t0_zstack =str(datetime.time(datetime.now()))
        self.t0_zstack.append(t0_zstack)

        writer.write_many_sample(self.matrix)
        # since it is only a volume, here the task waits until
        #  all the sample have been transmitted, then it changes a flag
        self.task.wait_until_done(float(1/frequency)*planes + .5)
        self.task.stop()
        # while(self.task.is_task_done() != True):
        #     pass
        # self.single_volume_done = True
        return self.t0_zstack

                                                                                               
 #---------------------------------------------------------------------------------------
class multi_DAQ_analogOut_shutter_rec(object):
    """ Class to control the light sheet acquisition. The DAQ controls plane galvo, 
    depth galvo, camera trigger, etl scan (analog channels). There are functions to acquire
    time lapses, volumes and time lapses of volumes. 
    """
    def __init__(self, device_name='Dev1', analog_out_channel=['ao0','ao1','ao2','ao3'],analog_in_channel=['ai1'], tasks_names=['analog_in','analog_read'] ):

        """ Constructor:
                - device_name = obvious;
                - channels = list with channels' names...
                    -'ao0' for image scan
                    -'ao1' for depth scan
                    -'ao2' for etl scan
                    -'ao3' for camera trigger
                    -'ai1' fir shutter rec
            The logic is: open the DEVICE, select the OUTPUT CHANNELS that send data,
            define a TASK that contains the functions to transmit and manipulate signals.
        """
        # self.device_name = device_name
        # self.channels = channels
        # self.task = nidaqmx.Task()

        self.device_name = device_name
        self.analog_in_channel= analog_in_channel
        self.analog_out_channel = analog_out_channel
        self.tasks_names = tasks_names
        
        # fill the channels
        if( analog_in_channel!=0):
            self.read_task = nidaqmx.Task(tasks_names[1])
            for i in range (len( self.analog_in_channel)):
              self.read_task.ai_channels.add_ai_voltage_chan(self.device_name
                                                    +'/'+self.analog_in_channel[i], terminal_config = nidaqmx.constants.TerminalConfiguration.RSE)
        
        if(analog_out_channel!=0):  
            self.analog_task = nidaqmx.Task(tasks_names[0])
             
            for i in range (len(self.analog_out_channel)):   
                self.analog_task.ao_channels.add_ao_voltage_chan(
                                                    self.device_name
                                                    +'/'+self.analog_out_channel[i])



        self.single_volume_done = False

    def stop(self):
        """ Stop the tasks.
            Close it (forget everything).
            Then open again.
        """
        # Analog_in channel 
        if(self.analog_in_channel!=0):
            for i in range (len(self.analog_in_channel)):
                if(self.analog_in_channel[i]=='ai1'):
                    self.read_task.stop()
                    self.read_task.close()
                    self.read_task = nidaqmx.Task(self.tasks_names[1])
                    self.read_task.ai_channels.add_ai_voltage_chan(self.device_name
                                                    +'/'+self.analog_in_channel[i], terminal_config = nidaqmx.constants.TerminalConfiguration.RSE)

        # Analog out channel 
        if(self.analog_out_channel!=0): 
            self.analog_task.stop()
            self.analog_task.close()
            self.analog_task = nidaqmx.Task(self.tasks_names[0])
            for i in range (len(self.analog_out_channel)):
            # add the active channels to the task
                self.analog_task.ao_channels.add_ao_voltage_chan(
                self.device_name+'/'+self.analog_out_channel[i])
            # default everything to 0
            if(len(self.analog_out_channel) == 2):
                self.analog_task.write([0, 0])
            else:
                self.analog_task.write([0, 0, 0, 0])







        # self.task.stop()
        # self.task.close()
        # self.task = nidaqmx.Task()
        # for i in range (len(self.channels)):
        #     # add the active channels to the task
        #     self.task.ao_channels.add_ao_voltage_chan(\
        #                         self.device_name+'/'+self.channels[i])
        #     # default everything to 0
        # if(len(self.channels) == 2):
        #     self.task.write([0, 0])
        # else:
        #     self.task.write([0, 0, 0, 0])

    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.analog_task.stop()
        self.analog_task.close()

        self.read_task.stop()
        self.read_task.close()

    def single_plane_acquisitions(self, frequency = 1.,\
                                  V_max = 1, V_min = -1):
        """ Acquire single planes (forever).
            - frequency = obvious
            - V_max = voltage when the light sheet is at the top
            - V_min = same, but at the bottom
        """
        assert(self.analog_out_channel == ['ao0', 'ao3']), \
                "\n Select only channels '\ao0\' & \'ao3\' for this modality"

        num_samples =  10**4 # default from the DAQ specifications

        self.t = np.linspace(0, 2/frequency, num_samples, dtype=np.float16)

        one_second = int(len(self.t)*frequency/2) # samples in one sec
        
        # triangular signal to scan the plane: every period 
        # (of the triang wave) scans 2 times (up and down)
        self.signal_1 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_1 = triang(int(num_samples))\
                     * (V_max - V_min)\
                     + V_min
        self.signal_1[:] = signal_1[:]

        # rect signal to define the trigger to the camera
        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t * frequency, duty = .1)+1)/2.
        signal_2[int(len(signal_2)*3/4+1):] = 0
        self.signal_2[:] = signal_2[:]

        # put them in a matrix to write with the task
        # (when write_many_samples is called)
        self.matrix = np.zeros((2, len(self.signal_1)))
        self.matrix[0,:] = self.signal_1
        self.matrix[1,:] = self.signal_2
        
        # CONTINUOUS means that will go on forever.
        # "num samples *2" because every channel transmits "samples", and we have 2 channels
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan=num_samples*2)

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=True)

        writer.write_many_sample(self.matrix)

    
    def single_plane_acquisitions_numbered(self, frequency = 1.,\
                                  V_max = 1, V_min = -1, num_images=10):
        """ Acquire a time lapse of one plane 
            - frequency = image acquisition frequency 
            - V_max = voltage when the light sheet is at the top
            - V_min = same, but at the bottom
            - num_images = images in time lapse 
        """
        assert(self.analog_out_channel == ['ao0', 'ao3']), \
                "\n Select only channels \ao0\' & \'ao3\' for this modality"

        num_samples =  10**4

        self.t = np.linspace(0, 2/frequency, num_samples, dtype=np.float16)

        one_second = int(len(self.t)*frequency/2) # samples in one sec
        
        
        self.signal_1 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_1 = triang(int(num_samples))\
                     * (V_max - V_min)\
                     + V_min
        self.signal_1[:] = signal_1[:]

        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t * frequency, duty = .1)+1)/2.
        signal_2[int(len(signal_2)*3/4+1):] = 0


        self.signal_2[:] = signal_2[:]

        self.matrix = np.zeros((2, len(self.signal_1)))
        self.matrix[0,:] = self.signal_1
        self.matrix[1,:] = self.signal_2
        
        
        # analog task for light sheet acquisition
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                    samps_per_chan=int(num_samples*num_images/2))
        # read task for light sheet acquisition
        self.read_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    source='ao/SampleClock',\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                    samps_per_chan=int(num_samples*num_images/2))
        # read task
        reader =nidaqmx.stream_readers.AnalogSingleChannelReader(self.read_task.in_stream)
        self.values_read = np.zeros(int(num_samples*num_images/2), dtype=np.float64)
        
        #analog task 
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream,auto_start=False)
       
        #starts tasks
        writer.write_many_sample(self.matrix)
        #time stamp 
        self.t0_im=[]
        t0_im =str(datetime.time(datetime.now()))
        self.t0_im.append(t0_im)
        #start tasks 
        self.read_task.start()
        self.analog_task.start() 
        reader.read_many_sample(self.values_read, number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,timeout=nidaqmx.constants.WAIT_INFINITELY) 
        

            
        
        # self.read_task.wait_until_done()
        # self.analog_task.wait_until_done()
        
        print('\nScan done. \n')
        
        return self.t0_im, self.values_read
                                  

    def acquisition_signals(self, frequency = 1., planes = 10, \
                                   V_plane_min = -.5, V_plane_max = .5, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28):
        """ Acquire time lapse of volumes until the stop button is pressed 
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
        """
        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 5.), \
                "\n Cannot send more than 5V to the ETL (channel 1)"
        assert(planes % 2 == 0), \
                "\n Cannot collect an odd number of planes"

        num_samples = 700*10**3 # int, from DAQ characteristics

        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)

        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        # it's a step function from Vmin to Vmax
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        self.signal_2 = np.zeros((len(steps_final)))*V_z_min
        for i in range(int(len(steps_final)/2)):
            if(steps_final[i]>.9):
                self.signal_2[i:] += z_step
        for i in range(int(len(steps_final)/2), len(steps_final)):
            if(steps_final[i]>.9):
                self.signal_2[i:] -= z_step

        # signal_3 is to change the ETL position
        # same as signal_2 but with different parameters
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step\

        self.signal_3[:] = (self.signal_2[:] + V_etl_offset) /V_etl_slope

        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2
        self.matrix[3,:] = self.signal_1

        # as beofore, define timings and samples length
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS)

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=True)

        writer.write_many_sample(self.matrix)
    
    def acquisition_signals_numbered(self, frequency = 1., planes = 10, \
                                   V_plane_min = -.5, V_plane_max = .5, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28,
                                   volumes=1):
        """ Acquire  a time lapse of volumes 
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
            - volumes = number of volumes to acquire
        """
        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 5.), \
                "\n Cannot send more than 5V to the ETL (channel 1)"
        assert(planes % 2 == 0), \
                "\n Cannot collect an odd number of planes"

        num_samples = 700*10**3 # it is considered an int -- check if enough!!!
        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)


        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        # self.signal_2 = np.zeros((len(steps_final)))*V_z_min
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] += z_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] -= z_step

        self.signal_2_b = np.ones((len(self.signal_0)))*V_z_min

        upflag = 1
        #inizia da mezzo duty
        for i in range(1, int((len(self.signal_1)/2))):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] += z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0
        upflag = 1
        for i in range(int(len(self.signal_1)/2), len(self.signal_1)):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] -= z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0





        # signal_3 is to change the ETL position
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step\

        self.signal_3[:] = (self.signal_2_b[:] + V_etl_offset) /V_etl_slope

        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2_b
        self.matrix[3,:] = self.signal_1

        # this is the only change from before.
        # FINITE acquisition, with different samps_per_chan
        # the logic of the "/2" is the same as for single planes:
        # the step functions of the ETL and z-galvo go up and down,
        # so a complete period covers 2 volumes
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(len(self.signal_0)*volumes/2))
                    # samps_per_chan=int(num_samples*volumes/2))
                    # samps_per_chan=volumes* int(len(self.t)/2))

        # read task for light sheet acquisition
        self.read_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    source='ao/SampleClock',\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                    samps_per_chan=int(len(self.signal_0)*volumes/2))
        # read task
        reader =nidaqmx.stream_readers.AnalogSingleChannelReader(self.read_task.in_stream)
        self.values_read = np.zeros(int(len(self.signal_0)*volumes/2), dtype=np.float64)

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=False)
        writer.write_many_sample(self.matrix)
        #time_stamp
        self.t0_zstack=[]
        t0_zstack =str(datetime.time(datetime.now()))
        self.t0_zstack.append(t0_zstack)

        #start tasks
        self.read_task.start()
        self.analog_task.start() 
        reader.read_many_sample(self.values_read, number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,timeout=nidaqmx.constants.WAIT_INFINITELY) 
        # print ('etl_offset ', V_etl_offset, 'etl_slope',V_etl_slope, 'V_etl_min', V_etl_min)
        # print('V_z_min', V_z_min, 'V_z_max', V_z_max, 'V_plane_max', V_plane_max, 'V_plane_min',V_plane_min, 'f',frequency, 'planes ',planes )

        return self.t0_zstack, self.values_read


    def single_volume(self, frequency = 1., planes = 10, \
                                   V_plane_min = -1, V_plane_max = 1, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28):
        """ Acquire a single volume.
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
        """
        self.single_volume_done = False


        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 4.99), \
                "\n Cannot send more than 5V to the ETL (channel 1)"

        num_samples = 700*10**3 # it is considered an int -- check if enough!!!
        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)


        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        self.signal_2 = np.ones((len(steps_final)))*V_z_min
        for i in range(int(len(steps_final)/2)):
            if(steps_final[i]>.9):
                self.signal_2[i:] += z_step
        for i in range(int(len(steps_final)/2), len(steps_final)):
            if(steps_final[i]>.9):
                self.signal_2[i:] -= z_step


        # signal 3  to ETL uses the fit parameters of the calibration ETL voltage 
        # galvo voltage.
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step

        self.signal_3[:] = (self.signal_2[:] + V_etl_offset) / V_etl_slope


        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2
        self.matrix[3,:] = self.signal_1

        # agai, the only change is in the sampls_per_chan
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                        samps_per_chan = planes * int(len(self.t)/2))
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=True)

        self.t0_zstack=[]
        t0_zstack =str(datetime.time(datetime.now()))
        self.t0_zstack.append(t0_zstack)

        writer.write_many_sample(self.matrix)
        # since it is only a volume, here the task waits until
        #  all the sample have been transmitted, then it changes a flag
        self.task.wait_until_done(float(1/frequency)*planes + .5)
        self.task.stop()
        # while(self.task.is_task_done() != True):
        #     pass
        # self.single_volume_done = True
        return self.t0_zstack
                                                                                                                                                                       

                                                                                                  
                                                                                                                                                      
 #---------------------------------------------------------------------------------------
class multi_DAQ_volume_shutter_rec(object):
    """ Class to control the light sheet acquisition. The DAQ controls plane galvo, 
    depth galvo, camera trigger, etl scan (analog channels). There are functions to acquire
    time lapses, volumes and time lapses of volumes. 
    """
    def __init__(self, device_name='Dev1', analog_out_channel=['ao0','ao1','ao2','ao3'],analog_in_channel=['ai1'], tasks_names=['analog_xyz_scan','analog_read_xyz_scan'] ):

        """ Constructor:
                - device_name = obvious;
                - channels = list with channels' names...
                    -'ao0' for image scan
                    -'ao1' for depth scan
                    -'ao2' for etl scan
                    -'ao3' for camera trigger
                    -'ai1' fir shutter rec
            The logic is: open the DEVICE, select the OUTPUT CHANNELS that send data,
            define a TASK that contains the functions to transmit and manipulate signals.
        """
        # self.device_name = device_name
        # self.channels = channels
        # self.task = nidaqmx.Task()

        self.device_name = device_name
        self.analog_in_channel= analog_in_channel
        self.analog_out_channel = analog_out_channel
        self.tasks_names = tasks_names
        
        # fill the channels
        if( analog_in_channel!=0):
            self.read_task = nidaqmx.Task(tasks_names[1])
            for i in range (len( self.analog_in_channel)):
              self.read_task.ai_channels.add_ai_voltage_chan(self.device_name
                                                    +'/'+self.analog_in_channel[i], terminal_config = nidaqmx.constants.TerminalConfiguration.RSE)
        
        if(analog_out_channel!=0):  
            self.analog_task = nidaqmx.Task(tasks_names[0])
             
            for i in range (len(self.analog_out_channel)):   
                self.analog_task.ao_channels.add_ao_voltage_chan(
                                                    self.device_name
                                                    +'/'+self.analog_out_channel[i])



        self.single_volume_done = False

    def stop(self):
        """ Stop the tasks.
            Close it (forget everything).
            Then open again.
        """
        # Analog_in channel 
        if(self.analog_in_channel!=0):
            for i in range (len(self.analog_in_channel)):
                if(self.analog_in_channel[i]=='ai1'):
                    self.read_task.stop()
                    self.read_task.close()
                    self.read_task = nidaqmx.Task(self.tasks_names[1])
                    self.read_task.ai_channels.add_ai_voltage_chan(self.device_name
                                                    +'/'+self.analog_in_channel[i], terminal_config = nidaqmx.constants.TerminalConfiguration.RSE)

        # Analog out channel 
        if(self.analog_out_channel!=0): 
            self.analog_task.stop()
            self.analog_task.close()
            self.analog_task = nidaqmx.Task(self.tasks_names[0])
            for i in range (len(self.analog_out_channel)):
            # add the active channels to the task
                self.analog_task.ao_channels.add_ao_voltage_chan(
                self.device_name+'/'+self.analog_out_channel[i])
            # default everything to 0
            if(len(self.analog_out_channel) == 2):
                self.analog_task.write([0, 0])
            else:
                self.analog_task.write([0, 0, 0, 0])







        # self.task.stop()
        # self.task.close()
        # self.task = nidaqmx.Task()
        # for i in range (len(self.channels)):
        #     # add the active channels to the task
        #     self.task.ao_channels.add_ao_voltage_chan(\
        #                         self.device_name+'/'+self.channels[i])
        #     # default everything to 0
        # if(len(self.channels) == 2):
        #     self.task.write([0, 0])
        # else:
        #     self.task.write([0, 0, 0, 0])

    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.analog_task.stop()
        self.analog_task.close()

        self.read_task.stop()
        self.read_task.close()

    

    
    def acquisition_signals_numbered(self, frequency = 1., planes = 10, \
                                   V_plane_min = -.5, V_plane_max = .5, \
                                   V_z_min = -1, V_z_max = 1,\
                                   V_etl_min = 0, V_etl_max = 4,
                                   V_etl_slope=0.5, V_etl_offset=1.28,
                                   volumes=1):
        """ Acquire  a time lapse of volumes 
            - frequency = obvious
            - planes = planes in the volume
            - V_plane_max = voltage when the light sheet is at the top
            - V_plane_min = same, but at the bottom
            - V_z_min = min voltage for the galvo controlling depth
            - V_z_max = max voltage for the galvo controlling depth
            - V_etl_min = min voltage for the ETL
            - V_etl_max = max voltage for the ETL
            - V_etl_slope = from calibration, slope to go from min to max V
            - V_etl_offset = again, from the linear calibration of ETL
            - volumes = number of volumes to acquire
        """
        z_step = (V_z_max-V_z_min) / planes
        assert(V_etl_max < 5.), \
                "\n Cannot send more than 5V to the ETL (channel 1)"
        assert(planes % 2 == 0), \
                "\n Cannot collect an odd number of planes"

        num_samples = 700*10**3 # it is considered an int -- check if enough!!!
        self.t = np.linspace(0, 2/frequency, int(num_samples/2/planes))

        one_second = int(len(self.t)*frequency/2)

        # signal_0 is for fast vertical scanning = sheet generation
        self.signal_0 = triang(int(num_samples/2/planes))\
                     * (V_plane_max - V_plane_min)\
                     + V_plane_min
        self.signal_0 = np.tile(self.signal_0, planes)


        # signal_1 is to trigger the camera acquisition
        self.signal_1 = 5 * (square(2 * np.pi * self.t * frequency, \
                                                            duty = .1)+1)/2.
        self.signal_1[len(self.signal_1)-1] = 0
        self.signal_1[int(len(self.signal_1)*3/4+1):] = 0
        self.signal_1 = np.tile(self.signal_1, planes)

        # signal_2 is to change the light sheet position ("z  galvo")
        steps = np.zeros(len(self.t))
        steps[int(len(self.t)/2):int(len(self.t)/2)+1] = 1.
        steps[int(len(self.t)-1)] = 1.
        steps_final = np.tile(steps, planes)

        # self.signal_2 = np.zeros((len(steps_final)))*V_z_min
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] += z_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_2[i:] -= z_step

        self.signal_2_b = np.ones((len(self.signal_0)))*V_z_min

        upflag = 1
        #inizia da mezzo duty
        for i in range(1, int((len(self.signal_1)/2))):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] += z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0
        upflag = 1
        for i in range(int(len(self.signal_1)/2), len(self.signal_1)):
            if(self.signal_1[i] >.9):
                if(upflag==0):
                    self.signal_2_b[i:] -= z_step
                    upflag = 1
            elif(self.signal_1[i] <.9):
                upflag = 0





        # signal_3 is to change the ETL position
        self.signal_3 = np.ones((len(steps_final))) * (V_etl_min)
        # for i in range(int(len(steps_final)/2)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] += etl_step
        # for i in range(int(len(steps_final)/2), len(steps_final)):
        #     if(steps_final[i]>.9):
        #         self.signal_3[i:] -= etl_step\

        self.signal_3[:] = (self.signal_2_b[:] + V_etl_offset) /V_etl_slope

        # the list seems strange because of forced connections in the DAQ pins
        # but it's working with the right outputs :)
        self.matrix = np.zeros((4, len(self.signal_0)))
        self.matrix[0,:] = self.signal_0
        self.matrix[1,:] = self.signal_3
        self.matrix[2,:] = self.signal_2_b
        self.matrix[3,:] = self.signal_1

        # this is the only change from before.
        # FINITE acquisition, with different samps_per_chan
        # the logic of the "/2" is the same as for single planes:
        # the step functions of the ETL and z-galvo go up and down,
        # so a complete period covers 2 volumes
        self.analog_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(len(self.signal_0)*volumes/2))
                    # samps_per_chan=int(num_samples*volumes/2))
                    # samps_per_chan=volumes* int(len(self.t)/2))

        # read task for light sheet acquisition
        self.read_task.timing.cfg_samp_clk_timing(rate = one_second,\
                    source='ao/SampleClock',\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,\
                    samps_per_chan=int(len(self.signal_0)*volumes/2))
        # read task
        reader =nidaqmx.stream_readers.AnalogSingleChannelReader(self.read_task.in_stream)
        self.values_read = np.zeros(int(len(self.signal_0)*volumes/2), dtype=np.float64)
        
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.analog_task.out_stream, auto_start=False)
        writer.write_many_sample(self.matrix)
        #time_stamp
        self.t0_zstack=[]
        t0_zstack =str(datetime.time(datetime.now()))
        self.t0_zstack.append(t0_zstack)

        # start tasks
        self.read_task.start()
        self.analog_task.start() 
        reader.read_many_sample(self.values_read, number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,timeout=nidaqmx.constants.WAIT_INFINITELY) 
        # print ('etl_offset ', V_etl_offset, 'etl_slope',V_etl_slope, 'V_etl_min', V_etl_min)
        # print('V_z_min', V_z_min, 'V_z_max', V_z_max, 'V_plane_max', V_plane_max, 'V_plane_min',V_plane_min, 'f',frequency, 'planes ',planes )

        return self.t0_zstack, self.values_read








class single_DAQ_shutter(object):
    """Class to control the shutter. It drives the shutter through a digital channel with 
       volteges 0 (closed), 5 (open).
       
    """

    def __init__(self, device_name='Dev1',  
                       digital_channels=['do0'],\
                       tasks_names=['digital_task']):

        """ Constructor:
                - device_name = obvious
                - digital_channels = list with digital channels' names
                -'do0' for shutter trigger
        """
        self.device_name = device_name
        self.digital_channels = digital_channels
        self.tasks_names = tasks_names
        
        # fill the channels
        if(digital_channels!=0):
            self.digital_task = nidaqmx.Task(tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
#-------------------------------------------------------------------------------                                                      
    def single_illumination(self, t_tot = 10.0, t_on = .001):
        assert(self.digital_channels == ['do0']), \
                "\n Select only channels \do0\' for this modality"
        
        """ Function to drive the shutter with single illumination. 
            t_tot= total time wave 
            duty= t_on/t_tot ----> t_on_shutter=t_tot*duty
            num_pulses=1
        """
    
                
        num_samples =  10**5
        num_pulses=1
        self.t_on=t_on
        self.duty = self.t_on/t_tot
        #time variable 
        self.t = np.linspace(0, t_tot, num_samples, dtype=np.float16)
        
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        
        #samples in one sec Changed here !!!!!
        samples_per_second = int(len(self.t))/t_tot
        # print('t_on',self.t_on, 'duty',self.duty, 't_tot', t_tot)
        # samples in one sec
        # one_second = int(len(self.t))
        
        # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))
        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        # plt.figure('One pulse')
        # plt.plot(self.t, self.trigger, 'bo-')
        # plt.grid()
        # plt.show()
        
        t_2P_single = str(datetime.time(datetime.now()))
        
        
        #start the task and when it is done stops it, close it and reinitialize
        self.digital_task.start()
        self.digital_task.wait_until_done(float(t_tot+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
   
        return  t_2P_single
#-------------------------------------------------------------------------------   
    def train_illumination(self, frequency = 1,t_on=0.005, num_pulses=1):
        """ Acquire a time lapse and the shutter opens (single illumination) when the time lapse starts 
            - frequency_im = image acquisition frequency 
            
            - num_images = obvious.
        """
        

        num_samples =  10**5
        
        self.t_on_2p= t_on
        # time variable calculated with the period of the im scan = 2* period
        #imaging !!! NB this is common for digital and analog
        self.t = np.linspace(0, 1/frequency, num_samples, dtype=np.float16)
        #------------------------Digital signal: 2P shutter---------------------
        # #time variable for 1 period of the scanning 
        self.t_tot=1/frequency
        # self.t_tot=1/frequency_im
        #duty calculated from time on and related to self.t so to the period of the scan
        self.duty_2p = self.t_on_2p/self.t_tot
        # self.duty_2p = self.t_on_2p/self.t_tot/2
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty_2p*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        # print('f_im',frequency_im,'f_scan', frequency_im/2, 'duty', self.duty_2p, 't_tot', self.t_tot)
        #samples in one sec digital channel.  !!!NB same as analog channel
        samples_per_second = int(len(self.t))/self.t_tot
       
        
        
       # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))

        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        self.t_2P_train = []
        t_2P_train =str(datetime.time( datetime.now()))
        self.t_2P_train.append(t_2P_train)
        
        self.digital_task.start()
        self.digital_task.wait_until_done(float(self.t_tot*num_pulses+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
        
        return self.t_2P_train
       
  
        
 
# ---------------------------------------------------------------------------------- 
   
    def train_illumination_old(self, frequency= 1., t_on = .001, num_pulses=10):
        
        """ Function to drive the shutter with a train of pulses. 
            f= frequency of square wave 
            duty= t_on*f ----> t_on_shutter=duty/f
            num_pulses=num repetions 
        """
        
        assert(self.digital_channels == ['do0']), \
                "\n Select only channels \do0\' for this modality"
                
        num_samples =  10**5
        
        t_tot=1/frequency
        self.t_on=t_on
        self.duty = self.t_on/t_tot
        #time variable 
        self.t = np.linspace(0, 1/frequency, int(num_samples), dtype=np.float16)
         # self.t = np.linspace(0, 1/frequency, int(num_samples/num_pulses), dtype=np.float16)
        #-------------------------------------------------
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        #single square wave with scipy
        # self.trigger = (2**32 - 1) * \
        #          (square(2 * np.pi * self.t*frequency , self.duty)+1)/2.
        #self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        #Total pulses train = repetitions of single square wave 
        self.trigger = np.tile(self.trigger, num_pulses)
        self.trigger = self.trigger.astype(np.uint32)
        #samples per second
        samples_per_second = int(len(self.t)*frequency)
        
        # samples in one sec
        # one_second = int(len(self.t)*frequency)
        
        # plot to check 
        # plt.figure('Pulses train')
        # # plt.plot(np.tile(self.t, num_pulses), self.trigger, 'bo-')
        # plt.plot(self.t, self.trigger, 'bo-')
        # plt.grid()
        # plt.show()
        
        
        # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))

        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        self.t_2P_train = []
        t_2P_train =str(datetime.time( datetime.now()))
        self.t_2P_train.append(t_2P_train)
        
        self.digital_task.start()
        self.digital_task.wait_until_done(float(t_tot*num_pulses+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
        
        return self.t_2P_train
#-------------------------------------------------------------------------------          
    
    def stop(self):
        """ Stop the task.
            Close it (forget eerything).
            Then open again.
        """
        self.digital_task.stop()
        self.digital_task.close()
 
        self.digital_task = nidaqmx.Task()
        if(self.digital_channels!=0):
            self.digital_task = nidaqmx.Task(self.tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
#-------------------------------------------------------------------------------    
    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.digital_task.stop()
        self.digital_task.close()
        
        
#-------------------------------------------------------------------------------        
        
