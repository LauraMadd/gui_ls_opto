import sys
from importlib import reload
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
from skimage.transform import resize
from PIL import Image
import time
from datetime import datetime
import pickle
import fnmatch
import os
from tabulate import tabulate
#our librariers
#import hands_on_image as hoi
import hoi_plus_click as hoi
reload(hoi)
from hoi_2 import acquisition_thread as hoi2
import ETL
import pyobj_galvo_for_threading as daq
reload(daq)
# import pyobj_shutter_rec_for_threading as shutter_server
# reload(shutter_server)
from SLMcontroller import player, hologram, hologramFft
import utility_functions as uf
reload(uf)
import lights_control_arduino as lights
reload(lights)
# import matrix_control_arduino as matrix
# reload(matrix)
import filters
reload(filters)
import pyqt_macros as pym
reload(pym)
if __name__=="__main__":
    class neurinvestigation(qtw.QMainWindow):

        def __init__(self):
            """ This is the constructor.
            It creates all the variables needed, the tabs, filled with
            buttons, labels, and editable inputs.
            When you read 'calibrate these', those are the parameters
            that should be calibrated from the results of the
            other calibration program.
            self.V_plane_max and min are the voltages that put the light-sheet
            at the top or bottom of the FOV.
            the acquisition freq. at line 144 is empirical, but it works.
            """
            super().__init__()
            self.title = 'Neurodev Lab'
            self.setWindowTitle('Setup Control')
            self.resize(686, 545)
            self.image_size=500
            self.tabs = qtw.QTabWidget(self)

            self.V_step = .01 # about 1.5 um default movement step in z

            # calibrate these:
            self.galvo_microns_per_volts_factor = 129.782   # slope planar calibration LS
            self.etl_offset = 1.4738 #  offset in volt from lens-galvo z calibration
            self.etl_slope =  0.588365 # slope adim from from lens-galvo zcalibration
            self.V_plane_min = -2.4  # voltage corresponding to lower  extrema FOV
            self.V_plane_max= 0.4 # voltage corresponding to upper extrema FOV
            
            #- static variables ------------------------------------------------
            self.etl_step = (self.V_step) / self.etl_slope
            self.etl_value = 2.56
            self.acquisition_z_step = 1. # not imporant, updates automatically

            # ATTRIBUTES
            self.lens = ETL.Opto(port='COM7')
            self.lens.connect()
            self.lens.current(0)
            self.lens.mode('analog')

            self.lights = lights.Lights_control()
            # self.matrix = matrix.Matrix_control()
            self.filters = filters.Filters()
            # Initialize class with camera and intractive image inside
            self.image_interaction = hoi2(\
                                    hoi.HandsOn(\
                                    np.random.rand\
                                    (self.image_size, self.image_size)), \
                                    ['zyla', 'AndorSDK3', 'Andor sCMOS Camera'],\
                                    self.image_size)
            # Initialize class to click on the image
            self.select_points = hoi.HandsOn(\
                                                np.random.rand(\
                                                self.image_size, self.image_size))

            self.selected_path = 'none'
            self.s_lasers='none'

            self.cam_flag = True
            self.cam_dim = 2048
            self.offset = self.cam_dim/2
            self.pix_size = 6.5
            self.M_2 = 9./300 #detection magnification
            self.volume_slider = 0 # keep track of the slice
            self.live_mode_on = False
            self.points = []# initialize list where selected points are saved
            self.z_value = 0
            self.z_min = 0
            self.z_max = 0
            self.z_min_set = 0
            self.z_max_set = 0
            self.num_planes = 10
            self.image_interaction.images_per_volume = self.num_planes
            self.single_volume = []
            # self.savedata_path = 'E:\\DATA\\Laura\\Ablation\\'
            self.savedata_path = 'E:\\DATA\\Bram\\2023_02_01'
            self.path_now = 'E:\\DATA\\Bram\\2023_02_01'
            # self.savedata_path = 'E:\\DATA\\Wieke\\2023_01_31'
            self.path_now = 'E:\\DATA\\Wieke\\2023_01_31'
            self.light_flag=0 #Initiaize flag for lights control
            self.holo_off=np.zeros((1152,1920),dtype=np.uint32)
            self.coords_holo=0
            self.acquisition_mode=''

            self.field_timing=[ 'start','time_stamp']
            self.dict_timing_2P_train={}
            self.values_timing_2P_train=[]

            self.dict_timing_2P_single={}
            self.values_timing_2P_single=[]


            self.dict_timing_im={}
            self.values_timing_im=[]

            self.time_stamp_2P_single=[]
            self.time_stamp_2P_train=[]
            self.time_stamp_im=[]


            self.field_timing_multic=[ 'green','blue']
            self.dict_timing_im_multic={}
            self.values_timing_im_blue=[]
            self.values_timing_im_green=[]



            self.dt_2P_single=[]
            self.dt_2P_train=[]



            #Class imaging thread
            self.imaging_thread = qtc.QThread()
            self.image_interaction.moveToThread(self.imaging_thread)
            self.imaging_thread.start()



            #class + thread volume multicolors 
            self.volume_scan_multicolor= daq.volume_scan_multicolor(lights=self.lights, filters=self.filters,  channels = ['ao0', 'ao1', 'ao2', 'ao3'] , task_name=['xyz_scan_multicolor'])
            self.xyz_scan_multicolor_thread = qtc.QThread()
            self.volume_scan_multicolor.moveToThread(self.xyz_scan_multicolor_thread)
            self.xyz_scan_multicolor_thread.start()


            #Class + thread for live + 2D time lapse
            self.plane_scan = daq.xy_scan(analog_out_channel=['ao0','ao3'],analog_in_channel=['ai1'],task_name=['analog_task_scan_xy', 'analog_read_xy'])
            # self.plane_scan = daq.xy_scan_shutter_rec(analog_channels=['ao0','ao3'],task_name=['digital_task_scan_xy','analog_task_scan_xy', 'analog_read_xy'])
            self.xy_scan_thread = qtc.QThread()
            self.plane_scan.moveToThread(self.xy_scan_thread)
            self.xy_scan_thread.start()

            self.acquisition_frequency = 1/(2048. + 200 + 5) * 25000
                                                    # 1/(H + window + 5) * vel
            self.line_exposure = 200/25000 # pixel / sec window/velocity
            self.acquisition_frequency = np.round(self.acquisition_frequency, 4)
            self.plane_scan.frequency_im = self.acquisition_frequency

            self.plane_scan.V_max = self.V_plane_max
            self.plane_scan.V_min =  self.V_plane_min

            #new variables for shutter in class xy_scan
            self.plane_scan.t_on_2p=0.001
            self.plane_scan.n_pulses=10
            self.plane_scan.planes=10



            #Class + thread for ETL in live mode and time lapse
            self.etl_scan = daq.single_scan_etl(channels='ao1', task_name = 'etl')
            self.etl_scan_thread = qtc.QThread()
            self.etl_scan.moveToThread(self.etl_scan_thread)
            self.etl_scan_thread.start()
            self.etl_scan.value = (self.etl_offset + self.z_value)/ self.etl_slope
            self.etl_scan.constant()
            self.etl_scan.value = self.etl_value

            #Class + thread to follow ETL with galvo ao2  in live mode and time lapse
            self.z_scan = daq.single_scan_galvo_z(channels ='ao2' , task_name = 'z')
            self.z_scan_thread = qtc.QThread()
            self.z_scan.moveToThread(self.z_scan_thread)
            self.z_scan_thread.start()
            self.z_scan.galvo.constant(self.z_value)

            #Class + thread for volumetric imaging
            self.volume_scan = daq.volume_scan(channels = ['ao0', 'ao1', 'ao2', 'ao3'] , task_name = ['xyz_scan'])
            # self.volume_scan =daq.volume_scan(analog_out_channel = ['ao0', 'ao1', 'ao2', 'ao3'] , analog_in_channel=['ai1'], tasks_names=['analog_xyz_scan','analog_read_xyz_scan'] )
            self.xyz_scan_thread = qtc.QThread()
            self.volume_scan.moveToThread(self.xyz_scan_thread)
            self.xyz_scan_thread.start()
            self.volume_scan.frequency = self.acquisition_frequency
            self.volume_scan.V_etl_offset = self.etl_offset

            #Class + thread for volumetric imaging+ shutter record
            self.volume_scan_shutter = daq.volume_scan_shutter(analog_out_channel = ['ao0', 'ao1', 'ao2', 'ao3'] , analog_in_channel=['ai1'], tasks_names=['analog_xyz_scan','analog_read_xyz_scan'] )
            self.xyz_scan_shutter_thread = qtc.QThread()
            self.volume_scan_shutter.frequency = self.acquisition_frequency
            self.volume_scan_shutter.V_etl_offset = self.etl_offset
            self.volume_scan_shutter.moveToThread(self.xyz_scan_shutter_thread)
            self.xyz_scan_shutter_thread.start()



            # Class+ thread shutter
            self.shutter = daq.shutter(digital_channels= ['do0' ] , task_name = 'shutter' )
            self.shutter_thread = qtc.QThread()
            self.shutter.moveToThread(self.shutter_thread)
            self.shutter_thread.start()

            self.shutter.t_tot=1
            self.shutter.frequency=4.
            #In common with shutter in xy scan
            self.shutter.t_on=0.001
            self.shutter.n_pulses=10
            self.list_xStart_zoom=[]
            self.list_yStart_zoom=[]
            self.list_dim_zoom=[]

            self.select_points.stack_index = self.z_min_set * self.galvo_microns_per_volts_factor
            #For SLM control
            # self.SLM=player.fromFile("Meadowlark_test_@sample.slm")
            self.SLM=player.fromFile("Meadowlark@sample_561nm.slm")
            self.cgh_path=''
            self.savepath_data="E:\\DATA\\"
            self.name_acquisition='stack'
            self.holo_name='cgh'
            self.name_for_acquisition = 'stack'
            self.single_volume = []
    # ------------------------------------------------------------------------------
            # TABS
    # #--------------------1 setup tab1:  Explore volume ----------------------------
            self.exploration_tab = qtw.QWidget()
            self.exploration_tab_layout = qtw.QGridLayout()
            self.exploration_tab.setLayout(self.exploration_tab_layout)
    #--------------------2 setup tab2:  Select points in volume +imaging  ----------
            self.volume_select_tab = qtw.QWidget()
            self.volume_select_tab_layout = qtw.QGridLayout()
            self.volume_select_tab.setLayout(self.volume_select_tab_layout)
    #--------------------3 setup tab3:  Select points in volume --------------------
            self.SLM_tab = qtw.QWidget()
            self.SLM_tab_layout = qtw.QGridLayout()
            self.SLM_tab.setLayout(self.SLM_tab_layout)

            self.tabs.addTab(self.exploration_tab, '2D Exploration')
            self.tabs.addTab(self.volume_select_tab, '3D points')
            self.tabs.addTab(self.SLM_tab, 'SLM <3')
            self.setCentralWidget(self.tabs)

            self.tabs.currentChanged.connect(self.tab_navigation)
            self.show()
    # ------------------------------------------------------------------------------
            # TABS FILLING
    # ------------------------------------------------------------------------------
            # TAB 0
    # ------------------------------------------------------------------------------
            self.planes_number = pym.InputWidget(label='Planes',
                                        default_input='10',
                                        functions=[self.planes_number_changed],
                                        tooltip='how many planes to acquire')
            #
            self.acquire_numbered_planes_button = pym.MyButton(
                            name='Time lapse',
                            functions=[self.terminate_2D_acquisition,
                            self.start_lasers,
                            self.image_interaction.TL_continuos_saving_acquisition,
                            self.get_time_stamp_imaging,
                            self.plane_scan.scan_numbered],
                            tooltip='acquire N planesS')


            self.live_save_button = pym.MyButton(name='Live+Save',
                    functions=[self.terminate_2D_acquisition, self.start_lasers,
                            self.live_on,  self.image_interaction.start_live_save,
                            self.get_time_stamp_imaging,
                            self.plane_scan.scan_numbered],
                    tooltip='live')


            self.open_camera_button = pym.MyButton(name='Live',
                    functions=[self.terminate_2D_acquisition, self.start_lasers,
                            self.live_on,  self.image_interaction.start_live,
                            self.plane_scan.scan],
                    tooltip='live')

            self.stop_button = pym.MyButton(name='Stop',
                    functions=[self.terminate_live, self.stop_lasers],
                    tooltip='stop live')

            self.lasers_button = pym.Choice(items=['Dark','Blue','Green','Reset',
                                'BF OFF','BF WHITE 25%','BF ON', 'Blue_Green'],
                                functions=[self.laser_mode],
                                tooltip='color')

            self.filter_reset_button=pym.MyButton(
                            name='Reset_F',
                            functions=[ self.reset_filters],
                            tooltip='Reset filters')


            self.stop_livesave_button = pym.MyButton(name='Stop LS.',
                    functions=[self.stop_livesave, self.stop_lasers],
                    tooltip='Stop')

            self.stop_2D_acquisition_button = pym.MyButton(name='Stop Acq.',
                    functions=[self.terminate_2D_acquisition, self.stop_lasers],
                    tooltip='Stop Acq.')

            # self.zoom_in_button = pym.MyButton(name='Zoom in',
            #         functions=[self.zoom],
            #         tooltip='zoom in')

            # self.zoom_out_button = pym.MyButton(name='Zoom OUT',
            #         functions=[self.forget_zoom],
            #         tooltip='zoom out')

            self.t_on_2p= pym.InputWidget(label='2P_on [s]',
                                    default_input='0.001',
                                    functions=[self.t_on_2p_changed],
                                    tooltip='2P pulse duration')

            self.n_pulses= pym.InputWidget(label='n pulses',
                                    default_input='10',
                                    functions=[self.n_pulses_changed],
                                    tooltip='2P pulse number')

            self.exposed_pixel_height = pym.InputWidget(label='H[pixels]',
                                    default_input='200',
                                    functions=[self.exposed_pixel_height_changed],
                                    tooltip='pixels of the scanning window')

            self.line_scan_speed = pym.InputWidget(label='V[pxls/ms]',
                                    default_input='20',
                                    functions=[self.line_scan_speed_changed],
                                    tooltip='Chip window scanning speed')

            self.frequency = pym.InputWidget(label='F[Hz]',
                                    default_input=str(np.round(self.acquisition_frequency, 2)),
                                    functions=0,
                                    tooltip='pixels of the scanning window',
                                    readonly = True)
            self.forward_button = pym.MyButton(name='z-->',
                                                functions=[self.forward],
                                                tooltip='move forward by one plane')

            self.backward_button = pym.MyButton(name='<--z',
                                    functions=[self.backward],
                                    tooltip='move backward by one plane')

            self.displace_z_value = pym.InputWidget(label='z now',
                                    default_input='0',
                                    functions=[self.set_z_pos],
                                    tooltip='[microns]')

            self.z_step = pym.InputWidget(label='dz',
                                    default_input='0',
                                    functions=[self.set_dz],
                                    tooltip='[microns]')

            self.file_path_panel0 = pym.MyButton(name='File path',
                                    functions=[self.pick_path],
                                    tooltip='pick path')

            self.remember_forward_button = pym.MyButton(name='Set z max',
                                    functions=[self.remember_z_max],
                                    tooltip='remember z')

            self.remember_backward_button = pym.MyButton(name='Set z min',
                                    functions=[self.remember_z_min],
                                    tooltip='remember z')

            self.z_max_set_value = pym.InputWidget(label='Saved Zmax',
                                    default_input=str(self.z_max_set),
                                    functions=0,
                                    tooltip='[microns]',
                                    readonly=True)
            self.z_min_set_value = pym.InputWidget(label='Saved Zmin',
                                    default_input=str(self.z_min_set),
                                    functions=0,
                                    tooltip='[microns]',
                                    readonly=True)

            self.freq_2P_set_value = pym.InputWidget(label='2P_freq',
                                    default_input=str(self.shutter.frequency),
                                    functions=[self.freq_2p_changed],
                                    tooltip='[Hz]')

            self.shutter_single_button = pym.MyButton(name='Shutter_single',
                                    functions=[self.get_time_stamp_2P_single,
                                            self.shutter.single_pulse],
                                    tooltip='single 2P pulse')
    

            self.shutter_train_button= pym.MyButton(name='Shutter_train',
                                    functions=[self.get_time_stamp_2P_train,
                                    self.shutter.train_illumination],
                                    tooltip='train 2P pulse')


            self.metadata_timelapse_button =  pym.Choice(items=['TL','TL+2P_single',\
                                            'TL+2P_train','z+2P_single', 'z+2P_train','TL_z_mc' ],
                                            functions=[self.metadata_timelapse],
                                            tooltip='metadata')

            self.save_time_lapse_button=pym.MyButton(name='Save live',
                                            functions=[self.save_timelapse_live],
                                            tooltip='Save time lapse live')


            # self.lens_close_button=pym.MyButton(name='close etl',
            #                                 functions=[self.close_ETL],
            #                                 tooltip='close ETL communication')
            
#------------------------------------------------------------------------------------------


            self.acquire_numbered_volumes_button_tab0 = pym.MyButton(name='TL volume',
                                    functions=[self.start_lasers, 
                                    self.update_saving,
                                    self.image_interaction.TL_zstack_continuos_saving_acquisition,
                                    self.get_time_stamp_imaging,
                                    self.send_signals_to_acquire_numbered_volumes],
                                    tooltip='acquire the defined number of volumes')


            self.stop_3D_acquisition_button_tab0 = pym.MyButton(name='Stop 3D acq.',
                            functions=[self.terminate_3D_acquisition,
                            self.stop_lasers],
                            tooltip='stop all 3D acq. operations')

            self.acquire_TL_volume_multic_2P = pym.MyButton(name=' TL_z_mc',
                                functions=[self.update_saving,
                                self.image_interaction.TL_zstack_continuos_saving_acquisition,
                                # self.get_time_stamp_imaging,
                                self.timelapse_volume_multicolor],
                                tooltip='acquire the defined number of volumes')



            
#------------------------------------------------------------------------------------------
            self.exploration_tab_layout.addWidget(\
                                        self.image_interaction.image, 0, 0, 16, 1)
            self.exploration_tab_layout.addWidget(self.open_camera_button, 0, 2)

            
            self.exploration_tab_layout.addWidget(self.stop_button, 0, 3)
            self.exploration_tab_layout.addWidget(self.filter_reset_button, 0,4)
            self.exploration_tab_layout.addWidget(self.lasers_button, 1, 2)
            self.exploration_tab_layout.addWidget(self.file_path_panel0, 1, 3)
            
           
                                                
            self.exploration_tab_layout.addWidget(self.planes_number, 2, 2)
            self.exploration_tab_layout.addWidget(\
                                        self.acquire_numbered_planes_button, 2, 3)
            self.exploration_tab_layout.addWidget(\
                                            self.stop_2D_acquisition_button, 2, 4)

            self.exploration_tab_layout.addWidget(self.live_save_button, 3, 2)
            self.exploration_tab_layout.addWidget(\
                                                self.stop_livesave_button, 3, 3)
            self.exploration_tab_layout.addWidget(self.save_time_lapse_button, 3, 4)
           
            self.exploration_tab_layout.addWidget(self.t_on_2p, 4, 2)
            self.exploration_tab_layout.addWidget(self.freq_2P_set_value,4, 3)
            self.exploration_tab_layout.addWidget(self.n_pulses, 4, 4)

            self.exploration_tab_layout.addWidget(self.shutter_single_button, 5, 2)
            self.exploration_tab_layout.addWidget(self.shutter_train_button, 5, 3)
            self.exploration_tab_layout.addWidget(
                                                self.metadata_timelapse_button, 5, 4)
            

            # self.exploration_tab_layout.addWidget(\
                                        # self.timelapse_singlepulse_button, 4, 3)
            # self.exploration_tab_layout.addWidget(\
                                        # self.timelapse_trainpulse_button, 4, 4)

            # self.exploration_tab_layout.addWidget(self.zoom_in_button, 5,2)
            # self.exploration_tab_layout.addWidget(self.zoom_out_button, 5, 3)
              
            self.exploration_tab_layout.addWidget(self.acquire_numbered_volumes_button_tab0, 6, 2)
            self.exploration_tab_layout.addWidget(self.stop_3D_acquisition_button_tab0, 6, 3)
            self.exploration_tab_layout.addWidget(self.acquire_TL_volume_multic_2P, 6, 4)
          

            self.exploration_tab_layout.addWidget(self.forward_button, 7, 3)
            self.exploration_tab_layout.addWidget(self.backward_button, 7, 2)
            # self.exploration_tab_layout.addWidget(self.delta_t_field, 7, 4)
            
            

            self.exploration_tab_layout.addWidget(self.displace_z_value, 8, 2)
            self.exploration_tab_layout.addWidget(self.z_step, 8, 3)
            # self.exploration_tab_layout.addWidget(self.lens_close_button, 8, 4)

            self.exploration_tab_layout.addWidget(
                                                self.remember_backward_button, 9, 2)
            self.exploration_tab_layout.addWidget(
                                                self.remember_forward_button, 9, 3)

            self.exploration_tab_layout.addWidget(self.z_min_set_value, 10, 2)
            self.exploration_tab_layout.addWidget(self.z_max_set_value, 10, 3)

            self.exploration_tab_layout.addWidget(
                                        self.exposed_pixel_height, 11, 2)
            self.exploration_tab_layout.addWidget(self.line_scan_speed, 11, 3)
            self.exploration_tab_layout.addWidget(self.frequency, 11, 4)
          
    # ------------------------------------------------------------------------------
            # TAB 1
    # ------------------------------------------------------------------------------
            self.take_volume_button =  pym.MyButton(name='Take a volume',
                                    functions=[self.start_lasers,
                                    self.image_interaction.start_single_volume_acquisition,
                                    self.take_single_volume],
                                    tooltip='acquire volumes')
            self.cutoff_IRfilter_on_button =  pym.MyButton(name='Filter IR',
                                    functions=[self.filters.filter_IRcutoff],
                                    tooltip='filter out IR')
            self.save_volume_button =  pym.MyButton(name='Save volume',
                                    functions=[self.save_volume],
                                    tooltip='save volume')
            self.file_path_panel1 = pym.MyButton(name='File path',
                                    functions=[self.pick_path],
                                    tooltip='pick path')
     
            self.stop_3D_acquisition_button = pym.MyButton(name='Stop 3D acq.',
                                    functions=[self.terminate_3D_acquisition,
                                    self.stop_lasers],
                                    tooltip='stop all 3D acq. operations')
            self.volumes_number = pym.InputWidget(label='Volumes #',
                                        default_input='10',
                                        functions=[self.volumes_number_changed],
                                        tooltip='number of volumes to acquire')

            self.acquire_numbered_volumes_button = pym.MyButton(name='Acquire volumes',
                                    functions=[self.start_lasers, self.update_saving,
                                    self.image_interaction.TL_zstack_continuos_saving_acquisition,
                                    self.send_signals_to_acquire_numbered_volumes],
                                    tooltip='acquire the defined number of volumes')



            self.acquire_volume_simple = pym.MyButton(name='Acquire volume',
                            functions=[self.start_lasers, self.update_saving,
                            self.image_interaction.zstack_continuos_saving_acquisition,
                            self.take_single_volume_simple],
                            tooltip='acquire volumes')


                         #newww
            self.acquire_TL_volume_multic = pym.MyButton(name=' TL_vol_col',
                            functions=[ self.update_saving,
                            self.image_interaction.TL_zstack_continuos_saving_acquisition,
                            self.get_time_stamp_imaging,
                            self.timelapse_volume_multicolor],
                            tooltip='acquire the defined number of volumes')
            
                        #newww
            self.delta_t_field=pym.InputWidget(label='Delta_t [s]',
                            default_input='30',
                            functions=[self.set_delta_t],
                            tooltip='delta t')



            self.z_plus_button = pym.MyButton(name='z++',
                                    functions=[self.slice_plus_one],
                                    tooltip='look at deeper slice')
            self.z_minus_button = pym.MyButton(name='z--',
                                    functions=[self.slice_minus_one],
                                    tooltip='look at more superficial slice')
            self.del_last_coords_button = pym.MyButton(name='DEL last click',
                                    functions=[self.select_points.forget_last_point],
                                    tooltip='delete last click from memory')
            self.del_all_coords_button = pym.MyButton(name='DEL all clicks',
                                    functions=[self.select_points.empty_coords_list],
                                    tooltip='delete last click from memory')
            self.z_max_field_tab2 = pym.InputWidget(label='z max [microns]',
                                        default_input='0',
                                        functions=[self.set_z_max],
                                        tooltip='MAX Z')
            self.z_min_field_tab2 = pym.InputWidget(label='z min [microns]',
                                        default_input='0',
                                        functions=[self.set_z_min],
                                        tooltip='MIN Z')
            self.num_planes = pym.InputWidget(label='Planes #',
                                        default_input='10',
                                        functions=[self.set_num_planes],
                                        tooltip='Number of planes')

            self.zoom_in_tab_1_button = pym.MyButton(name='Zoom in',
                                        functions=[self.zoom_tab_1],
                                        tooltip='magnify')
            self.zoom_out_tab_1_button = pym.MyButton(name='Zoom out',
                                        functions=[self.forget_zoom_tab_1],
                                        tooltip='full image')
            self.click_mode_button = pym.Choice(items=['Zoom','Coords'],
                            functions=[self.click_mode],
                            tooltip='Select the type of manipulation to perform on the stack')
            self.name_acquisition = pym.InputWidget(label='Filename',
                                        default_input='stack',
                                        functions=[self.set_acquisition_name],
                                        tooltip='insert file name')
            self.save_pix = pym.MyButton(name='Save pixmap',
                                    functions=[self.save_pixmap],
                                    tooltip='save the current image as tiff')

            self.metadata_zstack_button = pym.Choice(items=['z_stack','TL_zstack'],
                            functions=[self.metadata_zstack],
                            tooltip='save metadata')






            self.volume_select_tab_layout.addWidget(self.select_points, 0, 0, 14, 1)
            self.volume_select_tab_layout.addWidget(self.take_volume_button, 0, 2)
            self.volume_select_tab_layout.addWidget(self.cutoff_IRfilter_on_button, 0, 3)
            self.volume_select_tab_layout.addWidget(self.z_min_field_tab2, 1, 2)
            self.volume_select_tab_layout.addWidget(self.z_max_field_tab2, 1, 3)
            self.volume_select_tab_layout.addWidget(self.num_planes, 2, 2)
            self.volume_select_tab_layout.addWidget(self.z_minus_button, 3, 2)
            self.volume_select_tab_layout.addWidget(self.z_plus_button, 3, 3)
            self.volume_select_tab_layout.addWidget(self.file_path_panel1, 4, 2)
            self.volume_select_tab_layout.addWidget(self.save_volume_button, 4, 3)
            self.volume_select_tab_layout.addWidget(self.acquire_volume_simple, 5, 2)
            self.volume_select_tab_layout.addWidget(self.stop_3D_acquisition_button, 5, 3)
            self.volume_select_tab_layout.addWidget(self.volumes_number, 7, 2)
            self.volume_select_tab_layout.addWidget(self.acquire_numbered_volumes_button, 7, 3)
            self.volume_select_tab_layout.addWidget(self.delta_t_field, 8, 2)
            self.volume_select_tab_layout.addWidget(self.acquire_TL_volume_multic, 8, 3)
            self.volume_select_tab_layout.addWidget(self.click_mode_button, 9, 3)
            self.volume_select_tab_layout.addWidget(self.name_acquisition, 9, 2)
            self.volume_select_tab_layout.addWidget(self.save_pix, 10, 2)
            self.volume_select_tab_layout.addWidget(self.metadata_zstack_button, 10, 3)
            self.volume_select_tab_layout.addWidget(self.zoom_out_tab_1_button, 11, 2)
            self.volume_select_tab_layout.addWidget(self.zoom_in_tab_1_button, 11, 3)
            self.volume_select_tab_layout.addWidget(self.del_last_coords_button, 12, 2)
            self.volume_select_tab_layout.addWidget(self.del_all_coords_button, 12, 3)
    # ------------------------------------------------------------------------------
            # TAB 2
    # ------------------------------------------------------------------------------
            self.open_SLM_button = pym.MyButton(name='Open SLM',
                                    functions=[self.open_SLM],
                                    tooltip='turn on the Spatial Light Modulator')
            # self.save_path_button = pym.MyButton(name='Save path',
            #                         functions=[self.choose_folder],
            #                         tooltip='Choose folder for saving')
            self.load_cgh_button = pym.MyButton(name='Load CGH',
                                functions=[self.load_holo],
                                tooltip='Load Computer Generated Hologram on SLM')
            self.cgh_path_button = pym.MyButton(name='CGH_file',
                                functions=[self.choose_cgh],
                                tooltip='select the .npy CGH')
            self.name_holo = pym.InputWidget(label='Holo name',
                                        default_input='holoName',
                                        functions=[self.set_holo_name],
                                        tooltip='Filename')
            # self.T_affine_button = pym.MyButton(name='Calculate T affine',
            #                         functions=[self.calculate_Taffine],
            #                         tooltip='Calculate affine transformation')
            # self.save_T_affine_button = pym.MyButton(name='Save T affine',
            #                         functions=[self.save_Taffine],
            #                         tooltip='Save the calculated affine transformation')
            self.load_T_affine_button = pym.MyButton(name='Load T affine',
                                    functions=[self.load_Taffine],
                            tooltip='Load a previously calculated affine transformation')
            self.get_coord_input = pym.Choice(items=['Coords','Zoomed'],
                                functions=[self.get_coords],
                                tooltip='Get input coordinates')
            # self.cgh_exp = pym.InputWidget(label='exp cgh (ms)',
            #                             default_input='0.',
            #                             functions=[self.set_cgh_exp],
            #                             tooltip='Get coords')
            self.cgh_calculation = pym.MyButton(name='Point CGH',
                                    functions=[self.calculate_holo],
                                    tooltip='Calculate point CGH')
            self.cgh_fft_calculation = pym.MyButton(name='Ext. CGH',
                            functions=[self.calculate_fft_holo_convolution],
                                    tooltip='Calculate the CGH with FFT and conv')
            # self.cgh_calculation_gated = pym.MyButton(name='Gated CGH',
            #                         functions=[self.calculate_holo_gated],
            #                         tooltip='Gated CGH')
            self.r_input = pym.InputWidget(label='d [um]',
                                        default_input='0',
                                        functions=[self.set_radius],
                                        tooltip='d coords in um')
            self.get_mask_button = pym.MyButton(name='Get mask',
                                    functions=[self.get_mask],
                                    tooltip='get the SLM mask')
            
            self.load_mask_button = pym.MyButton(name='Load mask',
                                    functions=[self.load_mask],
                                    tooltip='Load mask')
                                 

            self.metadata_cgh_button = pym.MyButton(name='Save Metadata',
                                    functions=[self.metadata_cgh],
                                    tooltip='save metadata')

            self.SLM_tab_layout.addWidget(self.open_SLM_button, 1, 2)
            self.SLM_tab_layout.addWidget(self. cgh_path_button, 1, 3)
            self.SLM_tab_layout.addWidget(self.load_cgh_button, 1, 4)
            # self.SLM_tab_layout.addWidget(self.T_affine_button, 2, 2)
            # self.SLM_tab_layout.addWidget(self.save_T_affine_button, 2, 3)
            self.SLM_tab_layout.addWidget(self.load_T_affine_button, 2, 2)
            self.SLM_tab_layout.addWidget(self.get_coord_input, 2, 3)
            self.SLM_tab_layout.addWidget(self.cgh_calculation, 2, 4)
            # self.SLM_tab_layout.addWidget(self. save_path_button, 3, 2)
            # self.SLM_tab_layout.addWidget(self.cgh_exp, 3, 4)
            self.SLM_tab_layout.addWidget(self.name_holo, 3, 2)
       
            # self.SLM_tab_layout.addWidget(self.cgh_calculation_gated, 5, 5)

            self.SLM_tab_layout.addWidget(self.metadata_cgh_button, 3,3 )
            self.SLM_tab_layout.addWidget(self.r_input, 3, 4)
            self.SLM_tab_layout.addWidget(self. get_mask_button, 4, 2)
            self.SLM_tab_layout.addWidget(self.cgh_fft_calculation,4,3)
            self.SLM_tab_layout.addWidget(self.load_mask_button,4,4)
            

    # ------------------------------------------------------------------------------
    # METHODS
    # -------------------------Tab 0------------------------------------------------
    # ------------------------------------------------------------------------------
        def closeEvent(self, event):
            """ Function to close all the instruments
                When you click the X it shows a messatge (qtw.QMessageBox.question).
                Reply 'yes' to close it, 'no' to keep it open.

                If the user wants to close, then there are a series of ifs, that
                depend on which tab is currently seleted.
                Since different tab have different objects and threads open,
                the program checks all of them: if there is any running thread,
                it is terminated and close.
                e.g. line 501: if self.imaging_thread.isRunning():
                                    self.imaging_thread.terminate()
                    that is: if the imagin_thread thread is running,
                    then terminate it.
                For the objects, simply all the 'close' events that are coded in the
                libraries are invoked, and everything should properly close.

            """
            question = qtw.QMessageBox.question(self, 'Going away?', 'Really? </3',\
                                        qtw.QMessageBox.Yes | qtw.QMessageBox.No)
            event.ignore()
            if question == qtw.QMessageBox.Yes:
                if(self.cam_flag != None):
                    if(self.tabs.currentIndex()==0):
                        self.image_interaction.cam.close()
                        self.plane_scan.galvo.close()
                        
                        self.z_scan.galvo.close()
                        self.shutter.close()
                        self.etl_scan.galvo.close()
                        self.cam_flag = None
                        self.SLM.close()
                        self.lens.close()

                        # self.filters.close()
                    if(self.tabs.currentIndex()==1):
                        self.image_interaction.cam.close()
                        self.volume_scan.galvo.close()
                        # self.volume_scan_shutter.close()
                        self.lens.close()
                        self.SLM.close()
                        # self.filters.close()
                    if(self.tabs.currentIndex()==2):
                        self.image_interaction.cam.close()
                        self.volume_scan.galvo.close()
                        # self.volume_scan_shutter.close()
                        self.lens.close()
                        self.SLM.close()
                        self.filters.close()
                if self.imaging_thread.isRunning():
                    self.imaging_thread.terminate()
                if self.xy_scan_thread.isRunning():
                    self.xy_scan_thread.terminate()
                if self.etl_scan_thread.isRunning():
                    self.etl_scan_thread.terminate()
                if self.z_scan_thread.isRunning():
                    self.z_scan_thread.terminate()
                if self.xyz_scan_thread.isRunning():
                    self.xyz_scan_thread.terminate()
                # if self.xyz_scan_shutter_thread.isRunning():
                #     self.xyz_scan_shutter_thread.terminate()
                if self.shutter_thread.isRunning():
                    self.shutter_thread.terminate()
                self.lights.close()
                self.filters.close()
                event.accept()
    #-------------------------------------------------------------------------------
        def planes_number_changed(self):
            """ Function to change the  number of planes that
                will be acquired when using the button
                'self.acquire_numbered_planes_button'.
                It changes the variable 'planes' in the
                self.plane_scan object that controls the galvos,
                to that it will generate a finite number of triggers,
                corresponding to the number of planes.
                It is always the same z position.

            """
            self.plane_scan.planes = int(self.planes_number.input.text())
            # self.plane_scan.planes = planes
    #-------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def terminate_2D_acquisition(self):
            """ Function to stop  time lapse imaging in one plane.
                It stops all the scanning from the galvos called plane_scan,
                makes the flag 'acquire_flag' False
                (so that there is nothing live or collecting images).
                Then it clears th circular buffer (self.image_interaction.cam.mmc.clearCircularBuffer()).
                Finally it prints a message to tell the user that
                everything is closed.
            """
            self.plane_scan.stop()
            self.image_interaction.cam.acquire_flag = False
            time.sleep(1)
            self.image_interaction.cam.mmc.clearCircularBuffer()
            # print('\n Camera eyes closed (-.-)', flush=True)
    #-------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def terminate_live(self):
            """ Function to interrupt live mode.
                It stops all the scanning from the galvos,
                makes the flag 'stop_signal' and live_mode_on
                properly True and False,
                (so that there is nothing live or collecting images).
                Then it clears th circular buffer (self.image_interaction.cam.mmc.clearCircularBuffer()).
                Finally it prints a message to tell the user that
                everything is closed.

            """
            self.plane_scan.stop()
            self.image_interaction.stop_signal = True
            self.live_mode_on = False
            time.sleep(1)
            self.image_interaction.cam.mmc.clearCircularBuffer()
            print('\n Camera eyes closed (-.-)', flush=True)
    #------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def stop_livesave(self):
            self.plane_scan.stop()
            self.live_mode_on = False
            self.image_interaction.stop_signal = True
            time.sleep(1)
            self.image_interaction.cam.mmc.clearCircularBuffer()


    #-------------------------------------------------------------------------------
        def start_lasers(self):
            """ Function to open lights sources and to selcect the corresponding
                filter in front of the camera.
                Each if evaluates the light_flag and sends accordigly a command to
                light sources and Filters. The command itself is describing what
                is physicallly happening in the filters selection and in the
                shutters.
            """
            if(self.light_flag==0):
                print("All lights off")
                self.lights.dark()
            elif(self.light_flag==1):
                self.filters.filter_620()
                self.lights.green()
            elif(self.light_flag==2):
                self.filters.filter_520()
                self.lights.blue()
            # elif(self.light_flag==3):
            #     self.filters.filter_dark()
            #     self.lights.brightfield_0()
            elif(self.light_flag==4):
                self.filters.filter_dark()
                self.lights.brightfield_on_25()
            # elif(self.light_flag==5):
            #     self.filters.filter_dark()
            #     self.lights.brightfield_on_50()
            # elif(self.light_flag==6):
            #     self.filters.filter_dark()
            #     self.lights.brightfield_on_75()
            # elif(self.light_flag==7):
            #     self.filters.filter_dark()
            #     self.lights.brightfield_on_100()

            # elif(self.light_flag==8):
            #     self.filters.filter_dark()
            #     self.matrix.coordinate_shine(7,7)
            #     self.matrix.coordinate_shine(8,7)
            #     self.matrix.coordinate_shine(7,8)
            #     self.matrix.coordinate_shine(8,8)
            # elif(self.light_flag==9):
            #     self.filters.filter_dark()
            #     self.matrix.bf_off()
            elif(self.light_flag==10):
                self.filters.filter_520()
                self.lights.blue_green()
            
    #-------------------------------------------------------------------------------
        
        def reset_filters(self):
            self.filters.filter_reset()
    

    #-------------------------------------------------------------------------------
        def update_saving(self):
            """ This function updates the path used to save the data.
            The updated varbiale is self.image_interaction.cam.savedata_path.
            """
            self.image_interaction.cam.savedata_path = self.savedata_path
    #-------------------------------------------------------------------------------
        def live_on(self):
            """Function to set the live flag on.
            This flag is checked by the program at various stages.
            (def terminate_live(self),
            def zoom(self),
            def forget_zoom(self),
            zoom_tab_1,
            forget_zoom_tab_1,)
            In general, when anything turns the camera on,
            this flag becomes True, so that every object can
            access a simple flag that tells it that the camera is working.
            """
            self.live_mode_on = True

    #-------------------------------------------------------------------------------
        def stop_lasers(self):
            """ Function to close lights sources and to selcect the corresponding
                filter in front of the camera.
                if/else evaluates the value of value of the light_flag to allow to close
                the light switched on. Filters are moved to the reset position in
                both cases.
            """
            if(self.light_flag==2 or (self.light_flag==1) or (self.light_flag==10)):
                self.lights.dark()
            else:
               self.lights.brightfield_0()
            #    self.matrix.bf_off()
            self.filters.filter_reset()
    #-------------------------------------------------------------------------------
        def save_timelapse_live(self):
            """ Function to save the live timelapse. The time lapse is saved as single
                tiff files in the selected path inside a folder labeled with date and time
            """
            timestr = time.strftime("%Y%m%d-%H%M%S")
            path_now= self.savedata_path+'timelapse_'+ str(timestr)+"\\"
            os.mkdir(path_now)
            # n_images=(len(self.image_interaction.acquired_timelapse))
            n_images=self.plane_scan.planes
            print(n_images)
            for i in range(n_images):
                matrix_im=np.asarray(self.image_interaction.acquired_timelapse[i],\
                                                                    dtype=np.uint16)
                tif_im=Image.fromarray(matrix_im)
                tif_im.save(path_now+str(i).zfill(5)+'_'+str(self.name_acquisition.input.text())+'.tif')

            print('time lapse saved')
            self.image_interaction.acquired_timelapse=[]

    #-------------------------------------------------------------------------------
        def laser_mode(self):
            """ Function to control the lights sources. A string variable s is chosen
            by the user in the GUI panel trhough a combobox. Each possible entry
            of the variable s is associated with a light_flag. This light_flag
            corresponds to different light sources on. The light_flag values are used
            in start_laser and stop_lasers functions which are called in all the fuctions
            to start and stop the imaging.
            The descriptions of what each number means are in the 'print' statements.
            """
            self.s_lasers = str(self.lasers_button.currentText())
            print(  self.s_lasers, type(self.s_lasers))
            if(self.s_lasers=='Green'):
                print('561 nm laser On ')
                self.light_flag=1
            elif(self.s_lasers=='Blue'):
                print('488 nm laser On')
                self.light_flag=2
            elif(self.s_lasers=='Dark'):
                self.light_flag=0
            # elif(self.s_lasers=='BF 0%'):
            #     self.light_flag=3
            elif(self.s_lasers=='BF WHITE 25%'):
                self.light_flag=4
            # elif(self.s_lasers=='BF 50%'):
            #     self.light_flag=5
            # elif(self.s_lasers=='BF 75%'):
            #     self.light_flag=6
            # elif(self.s_lasers=='BF 100%'):
                # self.light_flag=7
            elif(self.s_lasers=='Reset'):
                self.filters.filter_reset()
            elif( self.s_lasers=='BF ON'):
                self.light_flag=8
            elif( self.s_lasers=='BF OFF'):
                self.light_flag=9  
            elif( self.s_lasers=='Blue_Green'):
                self.light_flag=10  

    #-------------------------------------------------------------------------------
        # def time_lapse(self):
        #     if(str(self.acquire_numbered_planes_button.currentText())=='Time lapse'):
        #         print(str(self.acquire_numbered_planes_button.currentText()) )
        #         self.terminate_2D_acquisition()
        #         self.start_lasers(), self.update_saving()
        #         time.sleep(1)
        #         self.image_interaction.continuos_saving_acquisition()
        #         time.sleep(1)
        #         self.plane_scan.scan_numbered()
        #
        #     if(str(self.acquire_numbered_planes_button.currentText())=='single pulse'):
        #         print(str(self.acquire_numbered_planes_button.currentText()) )
        #         self.terminate_2D_acquisition()
        #         self.start_lasers, self.update_saving()
        #         self.image_interaction.continuos_saving_acquisition()
        #         self.plane_scan.scan_numbered_single_pulse()
        #
        #     if(str(self.acquire_numbered_planes_button.currentText())=='train pulses'):
        #         print(str(self.acquire_numbered_planes_button.currentText()) )
        #         self.terminate_2D_acquisition()
        #         self.start_lasers, self.update_saving()
        #         self.image_interaction.continuos_saving_acquisition()
        #         self.plane_scan.scan_numbered_train_pulse()

    #-------------------------------------------------------------------------------
        def zoom(self):
            """ Function to zoom in a selected subregion.
            It does not work in live mode (thus the first if).
            It saves in variables (like self.yStart...) the coords
            of the selected region, so that it is possible, afterwards,
            if the users wants to click,to retrieve the correct coordinates
            in the whole picture.
            """
            if(self.live_mode_on == True):
                print('\nCan\'t zoom while in live mode', flush=True)
            else:
            #save xy coordinates of the upper-left corner and bottom right corner
            #of the zoom in pixels of the camera
                self.yStart =\
                self.image_interaction.image.yStart * self.cam_dim / self.image_size
                self.yEnd =\
                self.image_interaction.image.yEnd * self.cam_dim / self.image_size

                self.xStart =\
                self.image_interaction.image.xStart * self.cam_dim/self.image_size
                self.xEnd =\
                    self.image_interaction.image.xEnd*self.cam_dim/self.image_size
                # set zoom dimensions, the zoom is a square with max dimensions
                width = (self.yEnd - self.yStart)
                height = (self.xEnd - self.xStart)
                self.dim_roi = max(width, height)
                self.region_of_interest = self.image_interaction.resized[
                                    self.image_interaction.image.xStart:\
                                    self.image_interaction.image.xEnd,\
                                    self.image_interaction.image.yStart:\
                                    self.image_interaction.image.yEnd]
                resized = resize(self.region_of_interest, \
                                        (self.image_size,self.image_size),\
                                        anti_aliasing=True)
                self.image_interaction.image.update_image(resized)
    #-------------------------------------------------------------------------------
        def forget_zoom(self):
            """ This functions un-does the zoom,
            thus updating the image in self.image_interaction.image
            whit a full frame picture.
            """
            if(self.live_mode_on == True):
                print('\nCan\'t zoom out while in live mode', flush=True)
            else:
                resized = resize(self.image_interaction.resized, \
                                            (self.image_size,self.image_size),\
                                            anti_aliasing=True)
                self.image_interaction.image.update_image(resized)
    #-------------------------------------------------------------------------------
        def t_on_2p_changed(self):
            self.plane_scan.t_on_2p = float( self.t_on_2p.input.text())
            self.shutter.t_on = float( self.t_on_2p.input.text())
    #-------------------------------------------------------------------------------
        def get_time_stamp_imaging(self):

            #cancel info previous run

            self.time_stamp_im=[]

            #measure time with ms precision
            t0_im =str(datetime.time(datetime.now()))

            #save t0 imaging in list
            self.time_stamp_im.append(t0_im)

            print('time_stamp_im',self.time_stamp_im[0])
    #-------------------------------------------------------------------------------

        def get_time_stamp_2P_single(self):


            #measure time 2P with millisecond precision
            t_2P_single = str(datetime.time(datetime.now()))
            #save t 2P in list
            self.time_stamp_2P_single.append(t_2P_single)

            # print('time_stamp_2P_single', self.time_stamp_2P_single)

    #-------------------------------------------------------------------------------
        def calculation_dt_2P_single(self):
            start_time = datetime.strptime(self.time_stamp_im[0], '%H:%M:%S.%f')
            for i in range (len(self.time_stamp_2P_single)):
                end_time = datetime.strptime(self.time_stamp_2P_single[i], '%H:%M:%S.%f')
                diff = end_time - start_time
                self.dt_2P_single.append((diff.seconds * 1000) + (diff.microseconds / 1000))




    #-------------------------------------------------------------------------------
        def get_time_stamp_2P_train(self):
            #cancel info previous run
            self.dt_2P_train=[]
            self.time_stamp_2P_train=[]
            #measure time 2P with millisecond precision
            t0_2P_train = str(datetime.time(datetime.now()))
            #save t 2P in list
            self.time_stamp_2P_train.append(t0_2P_train)
            #Measure dt between imaging and pulse train
            start_time = datetime.strptime(self.time_stamp_im[0], '%H:%M:%S.%f')
            end_time = datetime.strptime(self.time_stamp_2P_train[0], '%H:%M:%S.%f')
            diff = end_time - start_time

            self.dt_2P_train=(diff.seconds * 1000) + (diff.microseconds / 1000)

            # print('time_stamp_2P_train', self.time_stamp_2P_train[0])


    #-------------------------------------------------------------------------------
        def n_pulses_changed(self):
            self.plane_scan.n_pulses =int( self.n_pulses.input.text())
            self.shutter.n_pulses =int( self.n_pulses.input.text())
    #-------------------------------------------------------------------------------
        def freq_2p_changed(self):
            self.shutter.frequency =float( self.freq_2P_set_value.input.text())

    #-------------------------------------------------------------------------------
        def exposed_pixel_height_changed(self):
            """Function to sets the value for the height h of the camera window which
            follows the light sheet (moved by the galvos xy). The height is in pixels
            Here the corresponding scanning frequency is calculated using also
            the scanning speed v (pixels/ms).
            The smaller this number, the higher optical sectioning, at the expenses
            of a slower acquisition. Usually the user should consider the
            intensity of the signal that they can reach.
            The formula for self.acquisition_frequency is empirical, but it works.
            The selected pixel heights must be within boundaries coming from the
            intrinsic characteristics of the camera (this is why we have the initial if).
            After that, all the parameters are updated. In particular
            self.plane_scan.frequency = self.acquisition_frequency
            tells the scanning galvos that the frequency is changed,
            so that the movement of the ligh-sheet line still matches
            the movement of the exposed pixels window on the camera chip,
            otherwise the camera would collect nothing.
            """
            if (int(self.exposed_pixel_height.input.text()) <20 or
                int(self.exposed_pixel_height.input.text()) > 1000):
                print('Exposed height out of bounds (must be 20<h<1000)',
                        flush=True)
                return None
            else:
                self.image_interaction.exposed_pixel_height = \
                                        int(self.exposed_pixel_height.input.text())
                self.acquisition_frequency = \
                        float(self.image_interaction.line_scan_speed)/\
                        (2048. + float(self.exposed_pixel_height.input.text()) + 5)
                self.line_exposure = float(self.exposed_pixel_height.input.text())/\
                                    float(self.image_interaction.line_scan_speed)
                self.acquisition_frequency = np.round(self.acquisition_frequency, 4)
                self.frequency.input.setText(str(self.acquisition_frequency))
                self.plane_scan.frequency_im = self.acquisition_frequency
    #-------------------------------------------------------------------------------
        def line_scan_speed_changed(self):
            """Function to set the scanning speed v.
            v is the speed at which the active pixels window is scanned
            through the camera chip.
            The corresponding scanning frequency is calculated.
            The selected v must be within boundaries coming from the
            intrinsic characteristics of the camera (this is why we have the initial if).
            After that, all the parameters are updated. In particular
            self.plane_scan.frequency = self.acquisition_frequency
            tells the scanning galvos that the frequency is changed,
            so that the movement of the ligh-sheet line still matches
            the movement of the exposed pixels window on the camera chip,
            otherwise the camera would collect nothigng.
            """
            if (int(self.line_scan_speed.input.text()) < 7 or
                int(self.line_scan_speed.input.text()) > 100):
                print('Line scan speed ut of bounds (must be 7<v<100)', flush=True)
                return None
            else:
                self.image_interaction.line_scan_speed = \
                                        int(self.line_scan_speed.input.text())*1000
                print(self.image_interaction.line_scan_speed)
                self.image_interaction.exposed_pixel_height = \
                                        int(self.exposed_pixel_height.input.text())
                # NEED TO DO BOTH, otherwise the camera would re-adjust height to
                # maintain same framerate automatically.
                self.acquisition_frequency = \
                        float(self.image_interaction.line_scan_speed)/\
                        (2048. + float(self.exposed_pixel_height.input.text()) + 5)
                self.line_exposure = float(self.exposed_pixel_height.input.text())/\
                                    float(self.image_interaction.line_scan_speed)
                self.acquisition_frequency = np.round(self.acquisition_frequency, 4)
                self.frequency.input.setText(str(self.acquisition_frequency))
                self.plane_scan.frequency_im = self.acquisition_frequency
    #-------------------------------------------------------------------------------
        def forward(self):
            """ Function to increase the z position in the sample starting from the
                value of self.V_step. This increment is added to z _value which is the V
                given to the galvo z. It has to be converted to the V input for
                the ETL. This conversion is done using the fit parameters from ETL-
                galvo z calibration. The corresponding z  in um position is updated on the
                GUI
            """
            #safety if to avoid galvo damage
            if(self.etl_value + self.etl_step >= 4.99):
                # print('max z')
                return None
            else:
                # print('z_value',self.z_value)
                #add increment to voltage value
                self.z_value += self.V_step
                #conversion to ETL voltages according to calibration parameters
                self.etl_value = (self.etl_offset + self.z_value)/ self.etl_slope
                #move ETL
                self.etl_scan.galvo.constant(self.etl_value)
                #move galvo a02
                self.z_scan.galvo.constant(self.z_value)
                # print('Voltage step',"%.2f" % self.V_step, 'um step','um step',"%.2f" %(self.galvo_microns_per_volts_factor * self.V_step) )
                #update z position in the sample using the fit parameters from
                # calibration galvoy-position
                self.displace_z_value.input.setText(str(np.round(
                            self.galvo_microns_per_volts_factor * self.z_value, 4)))
                print('ETL voltage:', self.etl_value, 'galvo a02 voltage:',self.z_value)
    #-------------------------------------------------------------------------------
        def backward(self):
            """ Function to increase the z position in the sample according to
                self.V_step. This increment is added to z _value which is the V
                given to the galvo z. It has to be converted to the V input for
                the ETL. This conversion is done using the fit parameters from ETL-
                galvo z calibration.The corresponding z  in um position is updated on the
                GUI
            """
            #safety if to avoid galvo damage
            if(self.etl_value - self.etl_step <= 0):
                # print('min z')
                return None
            else:
                #Add increment to vltage value
                self.z_value -= self.V_step
                #Conversion to ETL voltage using calibration parameters
                self.etl_value = (self.etl_offset + self.z_value) / self.etl_slope
                #move ETL
                self.etl_scan.galvo.constant(self.etl_value)
                #move galvo a02
                self.z_scan.galvo.constant(self.z_value)
                # print('Voltage step',"%.2f" % self.V_step, 'um step',"%.2f" % (self.galvo_microns_per_volts_factor * self.V_step) )
                #update z position in the sample using the fit parameters from
                # calibration galvoy-position
                self.displace_z_value.input.setText(str(np.round(
                            self.galvo_microns_per_volts_factor * self.z_value, 4)))
            print('ETL voltage:', self.etl_value, 'galvo a02 voltage:',self.z_value)


    #------------------------------------------------------------------------------
        def close_ETL(self):
            # self.etl_scan.galvo.close()
            self.lens.close()
    #-------------------------------------------------------------------------------
        def set_z_pos(self):
            """Function to change the z position in the sample.
            The z position given by the user in um is converted to Voltages and
            then given in input to move the galvo a02 . This voltage is also
            converted to the ETL voltages and then the ETL is moved.
            """
            #read from GUI the user input value and converts it in Voltages
            self.z_value=float(self.displace_z_value.input.text())/\
                                        self.galvo_microns_per_volts_factor
            #Conversion to ETL voltage using calibration parameters
            self.etl_value = (self.etl_offset + self.z_value) / self.etl_slope
            #move ETL
            self.etl_scan.galvo.constant(self.etl_value)
            #move galvo
            self.z_scan.galvo.constant(self.z_value)
            print( self.z_value)
    #-------------------------------------------------------------------------------
        def set_dz(self):
            """Function to define the z step in microns. It updates the varia
            ble V_step
            """
            # Read user input and convert it to volts
            self.V_step = float(self.z_step.input.text())/\
                                        self.galvo_microns_per_volts_factor
            print ('V step and dz step changed to:', "%.2f" % self.V_step,self.z_step.input.text() )
            print(self.V_step)


    
    #-------------------------------------------------------------------------------
    #new
        def set_delta_t(self):
            """Function to set the number of planes to acquire
            """
            #self.num_planes = int(self.num_planes.input.text())
            self.delta_t = int(self.delta_t_field.input.text())
            print(self.delta_t)
    #-------------------------------------------------------------------------------
        def pick_path(self):
            """ This function opens a window to select the path in which
            files will be saved.
            """
            temp= str(qtw.QFileDialog.getExistingDirectory(self,
                                                "Select Directory",self.savedata_path))
            self.savedata_path=temp+'/'
            print('Folder selected:',  self.savedata_path)
            self.image_interaction.cam.savedata_path = self.savedata_path
    #-------------------------------------------------------------------------------

        def remember_z_max(self):
            """Function to set the current z as z max.
            This function works with voltages and displays on the GUI
            the values in um
            """
            self.z_max_set = float(self.z_value)
            #conversion in um
            self.z_max_set_value.input.setText(str(np.round(
                        self.galvo_microns_per_volts_factor * self.z_max_set, 4)))
            #conversion in um
            self.z_max_field_tab2.input.setText(str(np.round(
                        self.galvo_microns_per_volts_factor * self.z_max_set, 4)))
            self.z_max = float(self.z_value)
            print( self.z_max_set_value)
    #-------------------------------------------------------------------------------

        def remember_z_min(self):
            """Function to set the current z as z max.
                this function works with voltages and displays on the GUI
                the values in um
            """
            self.z_min_set = float(self.z_value)
            #conversion in um
            self.z_min_set_value.input.setText(str(np.round(
                        self.galvo_microns_per_volts_factor * self.z_min_set, 4)))
            self.z_min_field_tab2.input.setText(str(np.round(
                        self.galvo_microns_per_volts_factor * self.z_min_set, 4)))
            self.z_min = float(self.z_value)
            print( self.z_min_set_value)#-------------------------------------------------------------------------------
    # -------------------------Tab 1------------------------------------------------
    # ------------------------------------------------------------------------------
        def take_single_volume(self):
            """ Function to acquire a single volume given frequency, n planes, z min
                z_max. The galvo z is driven with voltages obtained converting the z
                position into V using fit parameters of  the  calibration galvo
                voltage-position LS.
                The voltages to drive the ETL are calculated from the galvo Voltages
                using sing fit parameters of the  calibration ETL voltage-galvo
                voltage.
            """
            self.volume_slider = 0
            self.volume_scan.frequency = self.acquisition_frequency
            self.volume_scan.planes = int(self.num_planes.input.text())
            self.volume_scan.V_z_min = self.z_min_set # Voltage
            self.volume_scan.V_z_max = self.z_max_set # Voltage
            self.volume_scan.V_plane_min = self.V_plane_min
            self.volume_scan.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())

            self.volume_scan.V_etl_min = (self.z_min_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_etl_max = (self.z_max_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_elt_slope = self.etl_slope
            self.volume_scan.V_elt_offset  = self.etl_offset
            # self.volume_scan.V_etl_slope = self.etl_slope
            self.volume_scan.acquire_volume()
            while(self.image_interaction.volume_done != True):
                pass
            time.sleep(.5)
            self.select_points.update_image(\
                                self.image_interaction.acquired_volume_resized[0])
            self.image_interaction.volume_done = False
            # self.lights.dark()
            # self.filters.filter_reset()
            self.select_points.stack_index = self.z_min_set * self.galvo_microns_per_volts_factor
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor
    
    # ------------------------------------------------------------------------------
        def tab_navigation(self):
            """ Function called when moving throught tabs.
                Since different tab have different objects and threads open,
                the program checks all of them: if there is any running thread,
                it is terminated and close.
                e.g. line 948: if self.xyz_scan_thread.isRunning():
                                    self.xyz_scan_thread.terminate()
                    that is: if the xyz_scan_thread (that is scanning a volume
                    giving also triggers to the camera) thread is running,
                    then terminate it.
                For the objects, simply all the 'close' events that are coded in the
                libraries are invoked, and everything should properly close.
                Then it creates the right objects and threads for the specific tab.
                e.g.
                self.z_scan = daq.single_scan(channel_name = 'ao2', task_name = 'z')
                --- create the object that will control one galvo,
                the one that moves the beam in z
                self.z_scan_thread = qtc.QThread()
                --- creates a thread, that is a parallel process
                self.z_scan.moveToThread(self.z_scan_thread)
                --- puts the object in the thread
                self.z_scan_thread.start()
                --- starts the Thread
            n.b. all the thread-thingy is required so that more processes can happen
            at the same time, for example the scan, the trigger, the storing of image...
            outside of the "central pipe" of the GUI. In this way the GUI stays active during,
            for example, live mode, and allows the user to press the "stop" button when the live
            should be interrupted.

            """
            if(self.tabs.currentIndex()==0):

                # if self.xyz_scan_thread.isRunning():
                #     self.xyz_scan_thread.terminate()
                # self.volume_scan.close()
                # self.volume_scan = 0

                print("Tab 0 ")
                # self.plane_scan = daq.xy_scan(task_name=['digital_shutter','analog_scan'])
                # self.plane_scan = daq.xy_scan()
                # self.xy_scan_thread = qtc.QThread()
                # self.plane_scan.moveToThread(self.xy_scan_thread)
                # self.xy_scan_thread.start()



                # self.open_camera_button.clicked.connect(self.plane_scan.scan)
                #
                # self.live_save_button.clicked.connect(self.plane_scan.scan_numbered)

                # self.acquire_numbered_planes_button.clicked.connect(
                #                                     self.terminate_2D_acquisition)
                # self.acquire_numbered_planes_button.clicked.connect(self.start_lasers)
                # self.acquire_numbered_planes_button.clicked.connect(self.update_saving)
                # self.acquire_numbered_planes_button.clicked.connect(\
                #                 self.image_interaction.continuos_saving_acquisition)
                # self.acquire_numbered_planes_button.clicked.connect(
                #                                       self.plane_scan.scan_numbered)


                # self.timelapse_singlepulse_button.clicked.connect(self.terminate_2D_acquisition)
                # self.timelapse_singlepulse_button.clicked.connect(self.start_lasers)
                # self.timelapse_singlepulse_button.clicked.connect(self.update_saving)
                # self.timelapse_singlepulse_button.clicked.connect(\
                #                 self.image_interaction.continuos_saving_acquisition)
                # self.timelapse_singlepulse_button.clicked.connect(self.plane_scan.scan_numbered_single_pulse)



                # self.timelapse_trainpulse_button.clicked.connect(self.terminate_2D_acquisition)
                # self.timelapse_trainpulse_button.clicked.connect(self.start_lasers)
                # self.timelapse_trainpulse_button.clicked.connect(self.update_saving)
                # self.timelapse_trainpulse_button.clicked.connect(\
                #                 self.image_interaction.continuos_saving_acquisition)
    #             self.timelapse_trainpulse_button.clicked.connect(self.plane_scan.scan_numbered_train_pulse)
    #
    #             self.plane_scan.frequency_im = self.acquisition_frequency
    #             self.plane_scan.V_max = self.V_plane_max
    #             self.plane_scan.V_min =  self.V_plane_min
                #Class to control ETL through DAQ
                # self.etl_scan = daq.single_scan(channel_name = 'ao1',
                #                                 task_name = 'etl')
                # self.etl_scan_thread = qtc.QThread()
                # self.etl_scan.moveToThread(self.etl_scan_thread)
                # self.etl_scan_thread.start()
                # self.etl_scan.value = (self.etl_offset + self.z_value)/ self.etl_slope
                # self.etl_scan.constant()

                # Class to perform z scan with galvos
                # self.z_scan = daq.single_scan(channel_name = 'ao2', task_name = 'z')
                # self.z_scan_thread = qtc.QThread()
                # self.z_scan.moveToThread(self.z_scan_thread)
                # self.z_scan_thread.start()

            elif(self.tabs.currentIndex()==1):
                print ("Tab 1 ")
                #cancel threads...
                # if self.xy_scan_thread.isRunning():
                #     self.xy_scan_thread.terminate()
                # if(self.plane_scan != 0):
                #     self.plane_scan.close()
                #     self.plane_scan = 0
                # if self.etl_scan_thread.isRunning():
                #     self.etl_scan_thread.terminate()
                # if(self.etl_scan !=0):
                #     self.etl_scan.close()
                #     self.etl_scan = 0
                # if self.z_scan_thread.isRunning():
                #     self.z_scan_thread.terminate()
                # if(self.z_scan !=0):
                #     self.z_scan.close()
                #     self.z_scan = 0
                #... and create new ones for volume acquisition
                # if(self.volume_scan == 0):

    #                 self.volume_scan = daq.volume_scan()
    #                 self.xyz_scan_thread = qtc.QThread()
    #                 self.volume_scan.moveToThread(self.xyz_scan_thread)
    #                 self.xyz_scan_thread.start()
    #
    #                 self.volume_scan.frequency = self.acquisition_frequency
    #                 self.volume_scan.V_etl_offset = self.etl_offset
    #
    #             elif(self.xyz_scan_thread.isRunning() == True):
    #                 pass
    #
    #             self.single_volume = []

            else: pass
    # ------------------------------------------------------------------------------
        def save_volume(self):
            """ Function to save the acquired volume. The Volume is saved as single
                tiff files in the selected path inside a folder labeled with date and time
            """
            timestr = time.strftime("%Y%m%d-%H%M%S")
            path_now= self.savedata_path+'stack_'+ str(timestr)+"\\"
            os.mkdir(path_now)
            for i in range(len(self.image_interaction.acquired_volume)):
                matrix_im=np.asarray(self.image_interaction.acquired_volume[i],\
                                                                    dtype=np.uint16)
                tif_im=Image.fromarray(matrix_im)
                tif_im.save(path_now+str(i).zfill(5)+'_'+str(self.name_acquisition.input.text())+'.tif')
            print('volume saved')
    # ------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def terminate_3D_acquisition(self):
            """ Function to stop time lapse imaging in 3D.
            First the galvo scan is stopped.
            Then the flag that signals to the camera that the acquisition
            is not continuing is put to False.
            Then the buffer is cleared, to be sure that at the next acquisition
            the new series of images will not contain any old image.
            """
            self.volume_scan.stop()
            self.image_interaction.cam.acquire_flag = False
            time.sleep(2)
            self.image_interaction.cam.mmc.clearCircularBuffer()
            print('\n Camera eyes closed (-.-)')
    # ------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def terminate_3D_acquisition_tab0(self):
            """ Function to stop time lapse imaging in 3D.
            First the galvo scan is stopped.
            Then the flag that signals to the camera that the acquisition
            is not continuing is put to False.
            Then the buffer is cleared, to be sure that at the next acquisition
            the new series of images will not contain any old image.
            """
            self.volume_scan_shutter.stop()
            self.image_interaction.cam.acquire_flag = False
            time.sleep(2)
            self.image_interaction.cam.mmc.clearCircularBuffer()
            print('\n Camera eyes closed (-.-)')
    # ------------------------------------------------------------------------------
        def volumes_number_changed(self):
            """ When the user changes the number of volume that they want to acquire,
            this function is called and updates the variable
            self.volume_scan.volumes
            Inside the object that takes care of generating all the functions
            in the DAQ for data acquisition (galvo control, trigger, etl).
            """
            volumes = int(self.volumes_number.input.text())
            self.volume_scan.volumes = volumes
            self.volume_scan_shutter.volumes = volumes
     # ------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def send_signals_to_acquire_numbered_volumes_tab0(self):
            """ This functions will use the volume_scan object,
            that controls planar galvo, z galvo, etl, and trigger,
            to send signals to acquire a FIXED NUMBER of volumes.
            This means that the volume_scan parameters are updated
            following what the user put in and then
            self.volume_scan.acquire_volumes_numbered()
            is called: this sends FINITE signals
            (square wave for trigger, triang wave for scanning,
            step wave for etl and z scan)
            to perform a FINITE volumetric acquisition.
            The trigger tells the camera when to start, the triang
            moves linearly the planar galvo, the steps are syncronized
            to move the scanned plane, z by z in, the volume.
            """
            
            self.volume_scan_shutter.frequency = self.acquisition_frequency
            self.volume_scan_shutter.planes = int(self.num_planes.input.text())
            self.volume_scan_shutter.V_z_min = self.z_min_set # Voltage
            self.volume_scan_shutter.V_z_max = self.z_max_set # Voltage
            self.volume_scan_shutter.V_plane_min = self.V_plane_min
            self.volume_scan_shutter.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())
            
            self.volume_scan_shutter.V_etl_min = (self.z_min_set + self.etl_offset)/ \
                                            self.etl_slope
            self.volume_scan_shutter.V_etl_max = (self.z_max_set + self.etl_offset)/ \
                                            self.etl_slope
            self.volume_scan_shutter.V_etl_slope = self.etl_slope
            self.volume_scan_shutter.V_etl_offset  = self.etl_offset
            # self.volume_scan.V_etl_slope = self.etl_slope

            self.volume_scan_shutter.acquire_volumes_numbered()
            
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor





    # ------------------------------------------------------------------------------
        @qtc.pyqtSlot()
        def send_signals_to_acquire_numbered_volumes(self):
            """ This functions will use the volume_scan object,
            that controls planar galvo, z galvo, etl, and trigger,
            to send signals to acquire a FIXED NUMBER of volumes.
            This means that the volume_scan parameters are updated
            following what the user put in and then
            self.volume_scan.acquire_volumes_numbered()
            is called: this sends FINITE signals
            (square wave for trigger, triang wave for scanning,
            step wave for etl and z scan)
            to perform a FINITE volumetric acquisition.
            The trigger tells the camera when to start, the triang
            moves linearly the planar galvo, the steps are syncronized
            to move the scanned plane, z by z in, the volume.
            """
            
            self.volume_scan.frequency = self.acquisition_frequency
            self.volume_scan.planes = int(self.num_planes.input.text())
            self.volume_scan.V_z_min = self.z_min_set # Voltage
            self.volume_scan.V_z_max = self.z_max_set # Voltage
            self.volume_scan.V_plane_min = self.V_plane_min
            self.volume_scan.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())
            
            self.volume_scan.V_etl_min = (self.z_min_set + self.etl_offset)/ \
                                            self.etl_slope
            self.volume_scan.V_etl_max = (self.z_max_set + self.etl_offset)/ \
                                            self.etl_slope
            self.volume_scan.V_etl_slope = self.etl_slope
            self.volume_scan.V_etl_offset  = self.etl_offset
            # self.volume_scan.V_etl_slope = self.etl_slope

            self.volume_scan.acquire_volumes_numbered()
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor


    #----------------------------------------------------------------------------------------
        # @qtc.pyqtSlot()
        def timelapse_volume_multicolor_with_2P(self):
            """This function is useful for FUCCI imaging

            """
            self.volume_slider = 0
            self.volume_scan_multicolor.frequency = self.acquisition_frequency
            self.volume_scan_multicolor.planes = int(self.num_planes.input.text())
            self.volume_scan_multicolor.V_z_min = self.z_min_set # Voltage
            self.volume_scan_multicolor.V_z_max = self.z_max_set # Voltage
            self.volume_scan_multicolor.V_plane_min = self.V_plane_min
            self.volume_scan_multicolor.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())

            self.volume_scan_multicolor.V_etl_min = (self.z_min_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan_multicolor.V_etl_max = (self.z_max_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan_multicolor.V_elt_slope = self.etl_slope
            self.volume_scan_multicolor.V_elt_offset  = self.etl_offset
            self.volume_scan_multicolor.n_volumes=self.volume_scan.volumes
            self.volume_scan_multicolor.delta_t=self.delta_t
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor
            self.volume_scan_multicolor.acquire_volumes_multicolor()
#--------------------------------------------------------------------

        def timelapse_volume_multicolor(self):
            """This function is useful for FUCCI imaging

            """
            self.volume_slider = 0
            self.volume_scan.frequency = self.acquisition_frequency
            self.volume_scan.planes = int(self.num_planes.input.text())
            self.volume_scan.V_z_min = self.z_min_set # Voltage
            self.volume_scan.V_z_max = self.z_max_set # Voltage
            self.volume_scan.V_plane_min = self.V_plane_min
            self.volume_scan.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())

            self.volume_scan.V_etl_min = (self.z_min_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_etl_max = (self.z_max_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_elt_slope = self.etl_slope
            self.volume_scan.V_elt_offset  = self.etl_offset
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor
            for i in range(self.volume_scan.volumes):
                print('Acquisition volume'+ str(self.volume_scan.volumes))
                print('Green channel acquisition')
                self.filters.filter_620()
                self.lights.green()
                self.volume_scan.acquire_volume() 
                self.lights.dark()
                self.values_timing_im_green.append(self.volume_scan.t0_zstack)
                self.volume_scan.clear_timing_zstack()
                self.volume_scan.stop()
                print('Blue channel acquisition')
                self.filters.filter_520()
                self.lights.blue()
                self.get_time_stamp_imaging()
                self.volume_scan.acquire_volume()
                self.lights.dark()
                self.values_timing_im_blue.append(self.volume_scan.t0_zstack)
                self.volume_scan.clear_timing_zstack()
                self.volume_scan.stop()
                time.sleep(self.delta_t)



     #-----------------------------------------------------------------------------------------
        def take_single_volume_simple(self):
            """ Function to acquire a single volume given frequency, n planes, z min
                z_max. The galvo z is driven with voltages obtained converting the z
                position into V using fit parameters of  the  calibration galvo
                voltage-position LS.
                The voltages to drive the ETL are calculated from the galvo Voltages
                using sing fit parameters of the  calibration ETL voltage-galvo
                voltage.
            """
            self.volume_slider = 0
            self.volume_scan.frequency = self.acquisition_frequency
            self.volume_scan.planes = int(self.num_planes.input.text())
            self.volume_scan.V_z_min = self.z_min_set # Voltage
            self.volume_scan.V_z_max = self.z_max_set # Voltage
            self.volume_scan.V_plane_min = self.V_plane_min
            self.volume_scan.V_plane_max = self.V_plane_max
            self.acquisition_z_step = (self.volume_scan.V_z_max\
                                    - self.volume_scan.V_z_min)\
                                    / int(self.num_planes.input.text())

            self.volume_scan.V_etl_min = (self.z_min_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_etl_max = (self.z_max_set + self.etl_offset) / \
                                            self.etl_slope
            self.volume_scan.V_elt_slope = self.etl_slope
            self.volume_scan.V_elt_offset  = self.etl_offset
            # self.volume_scan.V_etl_slope = self.etl_slope
            self.volume_scan.acquire_volume()
            self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor
    

    # ------------------------------------------------------------------------------
        def slice_plus_one(self):
            """Function to navigate forward  the z stack acquired. It updates the variable
            volume slider which is the index of images in the z stack
            """
            #If to notify that you are outside the stack dimension
            if(self.volume_slider ==
                                    len(self.image_interaction.acquired_volume)-1):
                        print('bau')
            else:
                #update index of image
                self.volume_slider +=1
                print(self.volume_slider)
                #update image on the screen
                self.select_points.update_image(\
                                self.image_interaction.acquired_volume_resized[
                                                                self.volume_slider]\
                                /np.amax(self.image_interaction.acquired_volume_resized[
                                                                self.volume_slider]))
                #z position of the slice in microns
                self.select_points.stack_index=(self.volume_slider * \
                                self.acquisition_z_step + self.z_min_set) * \
                                                self.galvo_microns_per_volts_factor
                self.z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor

                print( 'stack_index',self.volume_slider, 'z position [um]',\
                        self.select_points.stack_index, 'z_step_um ',self.z_step_um)
    #-------------------------------------------------------------------------------
        def slice_minus_one(self):
            """Function to navigate backward the z stack acquired. It updates the variable
                volume slider which is the index of images in the z stack
            """
            #If to notify that you are outside the stack dimension
            if(self.volume_slider == 0): print('miao')
            else:
                #update index of image
                self.volume_slider -=1
                print(self.volume_slider)
                #update image on the screen
                self.select_points.update_image(\
                                self.image_interaction.acquired_volume_resized[
                                                                self.volume_slider])

                z_step_um = self.acquisition_z_step*self.galvo_microns_per_volts_factor
                #z position of the slice in microns
                self.select_points.stack_index=(self.volume_slider * \
                                self.acquisition_z_step + self.z_min_set) * \
                                                self.galvo_microns_per_volts_factor
                print( 'stack_index',self.volume_slider, 'z position [um]',\
                            self.select_points.stack_index,'z_step_um ',z_step_um)
    #-------------------------------------------------------------------------------
        def set_z_min(self):
            """Function to set z min. Reads the user input and converts it in voltages
            """
            # self.z_min_set = float(self.z_min.input.text())/\
            #                             self.galvo_microns_per_volts_factor
            self.z_min_set = float(self.z_min_field_tab2.input.text())/\
                                        self.galvo_microns_per_volts_factor

    #-------------------------------------------------------------------------------
        def set_z_max(self):
            """Function to set z max. Reads the user input and converts it in voltages
            """
            # self.z_max_set = float(self.z_max.input.text())/\
            #                         self.galvo_microns_per_volts_factor
            self.z_max_set = float(self.z_max_field_tab2.input.text())/\
                                    self.galvo_microns_per_volts_factor

    #-------------------------------------------------------------------------------
        def set_num_planes(self):
            """Function to set the number of planes to acquire
            """
            #self.num_planes = int(self.num_planes.input.text())
            self.image_interaction.images_per_volume = int(self.num_planes.input.text())
    #-------------------------------------------------------------------------------
        def zoom_tab_1(self):
            """ This function performs the zoom
            (ROI SELECTION)
            for the objects contained in the tab_1.
            It performs in the same way as the zoom function in tab0 but it select the
            ROI on the select.points object
            """
            if(self.live_mode_on == True):
                print('\nCan\'t zoom while in live mode')
            else:
                # numbers in space 500 by 500
                self.yStart = self.select_points.yStart
                self.yEnd = self.select_points.yEnd
                self.xStart = self.select_points.xStart
                self.xEnd = self.select_points.xEnd
                # set roi
                width = (self.yEnd - self.yStart)
                height = (self.xEnd - self.xStart)
                self.dim_roi = max(width,height)
                # select points is 500 x500
                self.region_of_interest = self.select_points.array[
                                    int(self.select_points.xStart):\
                                    int(self.select_points.xStart+self.dim_roi),\
                                    int(self.select_points.yStart):\
                                    int( self.select_points.yStart+self.dim_roi)]
                resized = resize(self.region_of_interest, \
                                        (self.image_size,self.image_size),\
                                        anti_aliasing=True)
                self.select_points.update_image(resized)
                print(self.xStart, self.yStart, self.dim_roi)
    #-------------------------------------------------------------------------------
        def forget_zoom_tab_1(self):
            """Function to forget zoom_tab_1
            """
            if(self.live_mode_on == True):
                print('\nCan\'t zoom out while in live mode')
            else:
                resized = resize( self.image_interaction.acquired_volume_resized[
                                                                self.volume_slider], \
                                            (self.image_size,self.image_size),\
                                            anti_aliasing=True)
                self.select_points.update_image(resized)
                print('zomm_out')
    #-------------------------------------------------------------------------------
        def click_mode(self):
            """ Function to choose if the mouse click is associated to zoom or selection
                of points. The combo button choice changes the value of a string variable
                s and consequently a flag variable used in hands_on_image class.
                if coords the used can click and select coordinates.
                if zoom, the user can click and drag to perform a zoom in a selected ROI.
            """
            s = str(self.click_mode_button.currentText())
            if(s=='Coords'):
                self.select_points.drag_flag = False
            else:
                self.select_points.drag_flag = True
    #-------------------------------------------------------------------------------
        def set_acquisition_name(self):
            """Function to append file name to saved data
            """
            self.name_for_acquisition=self.name_acquisition.input.text()
            print('Acquisition name set')
    #---------------------------------------------SLM Tab---------------------------
    #-------------------------------------------------------------------------------
        def open_SLM(self):
            """ Function to open the SLM
            """
            self.SLM.startLoop()
            print('SLM ready')
    #------------------------------------------------------------------------------
        def choose_folder(self):
            """Function to choose a directory where we want to save data
            """
            temp = str(qtw.QFileDialog.getExistingDirectory(
                                            self, "Select Directory","E:\\DATA\\"))
            self.savepath_data=temp+'/'
            print('Folder selected:', self.savepath_data)
    #------------------------------------------------------------------------------
        def load_holo(self):
            """Function to load in the SLM the CGH as a npy file.
            Time sleep to allow LC to orient.
            """
            self.pattern = np.load(self.cgh_path)
            self.SLM.setPhase(self.pattern)
            time.sleep(.4)
            print('Hologram loaded on SLM')
    #------------------------------------------------------------------------------
        def choose_cgh(self):
            """Function to choose a CGH as npy file
            """
           
            temp= qtw.QFileDialog.getOpenFileName(None,"Select CGH",
                                            "C:\\CGH_patterns\\")
            self.cgh_path=temp[0]
            print('CGH selected ', self.cgh_path)
           
    #------------------------------------------------------------------------------

        def set_holo_name(self):
            """Function to set a name. This name will be appended to the following
            saved variables:
            -CGH,
            -coords CGH and intermediate coords.
            """
            self.holo_name = self.name_holo.input.text()
            print('files names set')
    #------------------------------------------------------------------------------
        def calculate_Taffine(self):
            """Function to calculte affine transform. This has to be updated. Currently
            the Affine T is calculated beforehand and then loaded as a matrix
            """
            self.load_Taffine_flag = 0
            n_points = 12
            coord_ref = np.ones((3, n_points))
            coord_ref [0,:] = np.linspace(-90, 100.0, n_points)
            coord_ref [1,:] = np.linspace(-80, 100.0, n_points)
            coord_ref [2,:] = np.linspace(-20, 20, n_points)
            # NOT GOOD...
            lens = ETL.Opto(port='COM24')
            lens.connect()
            lens.current(0)
            lens.mode('analog')
            scan = self.galvo.DAQ_analogOut(channel_name='ao1')
            defocus_voltage = -0.016 * coord_ref[2, :] +2.7
            patterns = fnmatch.filter(os.listdir(self.cgh_path), '*.npy')
            image = np.zeros((n_points,self.cam_dim,self.cam_dim), dtype=np.uint16)
            centers = np.zeros((3, n_points))
            coord_moving = np.ones((4, n_points), dtype=np.float64)
            for i in range(len(patterns)):
                name_holo = patterns[i]
                print(name_holo)
                #Load Pattern on SLM
                holo=np.load(self.cgh_path+name_holo)
                print ("setting phase")
                self.SLM.setPhase(holo)
                time.sleep(.5)
                scan.constant(defocus_voltage[i])
                time.sleep(.1)
    #---------------------- Call function T affine -----------------------------

            self.T_affine = uf.Find_Taffine_z(coord_ref,matrix_im,self.planes)
            print('Affine transformation ready!')
    #-------------------------------------------------------------------------------

        def save_Taffine(self):
            """Function to save the affine transform as a npy file.
            """
            pickle.dump(self.T_affine, open(self.savepath_data+"T_affine.p", "wb"))
            print('Affine transformation saved')
    #-------------------------------------------------------------------------------
        def load_Taffine(self):
            """Function to load an  affine transform as a npy file from a dialog window.
            When the affine T is loaded a flag variable is set to 1
            """
            temp = qtw.QFileDialog.getOpenFileName(None,"Select T_affine",
                                                    "E:\\DATA\\")
            self.Taffine_path = temp[0]
            self.T_affine=np.load(self.Taffine_path)
            self.load_Taffine_flag = 1
            print('Affine transformation loaded')
    #-------------------------------------------------------------------------------
        def get_coords(self):
            """Function to save a set of selected coordinates. These coords will be the
            input for CGH calculation.
            This function is diveded in two main block executed as else if according
            to the value of the string s gave from the user.
            One block saves coords selected from a 500x500 pixels preview image,
            the other one from a zoomed subregion of the 500x500 pixel preview image.
            Coords set are saved as npy file in a subfolder (in the save path0 labelled
            with the date and time.
            """


            #Calculation of coords selected without zoom
            if(str(self.get_coord_input.currentText())=='Coords'):
                print('if')
                #data_path to save the coords
                timestr = time.strftime("%Y%m%d-%H%M%S")
                self.path_now = self.savedata_path+'coords_'+ str(timestr)+"\\"
                os.mkdir(self.path_now)
                print('selected',self.select_points.coords)
                #coords as matrix
                selected_coords = np.asarray(self.select_points.coords, dtype=np.float)
                #conversion from image dysplayed  (500x 500 pixels) to real size camera , cartesian framework
                self.coords = np.zeros((selected_coords.shape), dtype=np.float)
                self.coords[:,0] = selected_coords[:,1]*self.cam_dim/self.image_size
                self.coords[:,1] = selected_coords[:,0]*self.cam_dim/self.image_size
                self.coords[:,2] = selected_coords[:,2]
                #Conversion to the sample plane in microns and to cartesian framework
                self.coords[:,0] = (self.coords[:,0]-self.offset)*self.pix_size*self.M_2
                self.coords[:,1] = (self.offset-self.coords[:,1])*self.pix_size*self.M_2
                print('Conversion in um in sample plane',self.coords)
                np.save(self.path_now+'coords_um_sample'+self.name_holo.input.text(),self.coords)
                #Coords put in a suitable array to apply T affine
                coords_4affine=np.ones((4,self.coords.shape[0]),dtype=float)
                coords_4affine[:3,:]=np.transpose(self.coords)
                print('Transposed coords',coords_4affine, 'shape', coords_4affine.shape)
                # Application T_affine
                coords_transf=np.zeros((3,self.coords.shape[0]),dtype=float)
                coords_transf=np.dot(self.T_affine ,coords_4affine)
                np.save(self.path_now+'Affine_T_coords_'+self.name_holo.input.text(),coords_transf)
                # Transformations due to setup layout
                # Flip up-down
                coords_flip=np.copy(coords_transf)
                coords_flip[1,:]=coords_transf[1,:]*(-1.0) # flipy
                print('flipped coords',coords_flip, 'shape', coords_flip.shape)
                np.save(self.path_now+'Flipped_coords_'+self.name_holo.input.text(),coords_flip)
                #Apply rot matrix -90
                coords_input=np.copy(coords_flip)
                rot_matrix=np.zeros((2,2))
                rot_matrix[0,1]=1
                rot_matrix[1,0]=-1
                coords_rot=np.dot(rot_matrix,coords_flip[:2,:])
                print('rotated coords',coords_rot, 'shape', coords_rot.shape)
                np.save(self.path_now+'Rotated_coords_'+self.name_holo.input.text(),coords_rot)
                #Input coords for CGH calculation
                coords_input[0,:]=coords_rot[0,:]
                coords_input[1,:]=coords_rot[1,:]
                print('coords_input',coords_input)
                #Coords for CGH calculation transposed +saved
                self.coords_holo=np.transpose(coords_input)
                print('final',self.coords_holo)
                np.save(self.path_now+'Final_coords_'+self.name_holo.input.text(),self.coords_holo)
            else:
                print('else')
                timestr = time.strftime("%Y%m%d-%H%M%S")
                self.path_now= self.savedata_path+'zoom_coords_'+ str(timestr)+"\\"
                os.mkdir(self.path_now)
                print('selected zoomed coords',self.select_points.coords)
                #coords as a matrix
                selected_coords = np.asarray(self.select_points.coords, dtype = np.float)
                self.coords=np.zeros((selected_coords.shape),dtype=np.float)
                #conversion from image displaced to roi , cartesian framework
                self.coords[:,0]=selected_coords[:,1]*self.dim_roi/self.image_size
                self.coords[:,1]=selected_coords[:,0]*self.dim_roi/self.image_size
                self.coords[:,2]=selected_coords[:,2]
                print('Conversion to roi space',self.coords)
                # add off set due to roi
                self.coords[:,0]=self.coords[:,0]+self.yStart
                self.coords[:,1]=self.coords[:,1]+self.xStart
                print('Conversion to add offset',self.coords)
                # conversion from image displayed to camera spaace
                self.coords[:,0]=self.coords[:,0]*self.cam_dim/self.image_size
                self.coords[:,1]=self.coords[:,1]*self.cam_dim/self.image_size
                print('Conversion in camera pixels',self.coords)
                #Conversion to the sample plane in microns
                self.coords[:,0]=(self.coords[:,0]-self.offset)*self.pix_size*self.M_2
                self.coords[:,1]=(self.offset-self.coords[:,1])*self.pix_size*self.M_2
                print('Conversion in um in sample plane',self.coords)
                np.save(self.path_now+'coords_um_sample'+self.name_holo.input.text(),self.coords)
                #Coords put in a suitable array to apply T affine
                coords_4affine=np.ones((4,self.coords.shape[0]),dtype=float)
                coords_4affine[:3,:]=np.transpose(self.coords)
                print('Transposed coords',coords_4affine, 'shape', coords_4affine.shape)
                # Application T_affine
                coords_transf=np.zeros((3,self.coords.shape[0]),dtype=float)
                coords_transf=np.dot(self.T_affine ,coords_4affine)
                print('transf coords',coords_transf, 'shape', coords_transf.shape)
                np.save(self.path_now+'Affine_T_coords_'+self.name_holo.input.text(),coords_transf)
                # Transformations due to setup layout
                # Flip up-down
                coords_flip=np.copy(coords_transf)
                coords_flip[1,:]=coords_transf[1,:]*(-1.0) # flipy
                print('flipped coords',coords_flip, 'shape', coords_flip.shape)
                np.save(self.path_now+'Flipped_coords_'+self.name_holo.input.text(),coords_flip)
                #Apply rot matrix -90
                coords_input=np.copy(coords_flip)
                rot_matrix=np.zeros((2,2))
                rot_matrix[0,1]=1
                rot_matrix[1,0]=-1
                coords_rot=np.dot(rot_matrix,coords_flip[:2,:])
                print('rotated coords',coords_rot, 'shape', coords_rot.shape)
                np.save(self.path_now+'Rotated_coords_'+self.name_holo.input.text(),coords_rot)
                #Input coords for CGH calculation
                coords_input[0,:]=coords_rot[0,:]
                coords_input[1,:]=coords_rot[1,:]
                print('coords_input',coords_input)
                #Coords for CGH calculation transposed+save
                self.coords_holo=np.transpose(coords_input)
                print('final',self.coords_holo)
                np.save(self.path_now+'Final_coords_'+self.name_holo.input.text(),self.coords_holo)
    #-------------------------------------------------------------------------------
        def set_cgh_exp(self):
            """Function to the exposure of the CGH series in ms. This exposure is
            then converted in s
            """
            self.exp_time = float(self.cgh_exp_input.text())
            print("cgh exposure set ",self.exp_time, " ms\n")
            self.duration_seq = self.exp_time*2/1000
            print("duration set ", self.duration_seq, " s\n")
    #-------------------------------------------------------------------------------
        def calculate_holo(self):
            """Function to calculate a point cloud CGH
            """
            # definition of intensities vector, here all points have same intensity
            intensities=np.ones(self.coords_holo.shape[0])
            #vector to list
            self.int_list=intensities.tolist()
            #Function to define the CGH
            holo_target=hologram(points = self.coords_holo,intensities=self.int_list,\
                        aberrations = None,playerInst = self.SLM,sourceIndex = 0,\
                        apertureIndex=0)
            #Function to calculate the CGH, here we use CSWGS as algo
            holo_target.compute("CSWGS",100,0.5)
            holo_target.wait()
            #Save CGH as phase pattern in the above chosen directory
            np.save(self.path_now+'phase_'+self.name_holo.input.text(),holo_target.phase)
            # Load Cgh on SLM
            print ("Setting phase")
            self.SLM.setHologram(holo_target)
            #optional
            # set the CGH exposere time
            # time.sleep(5.0)
            # #blank image on SLM ofter exposure
            # self.SLM.setPhase(self.holo_off)
    #-------------------------------------------------------------------------------

        def calculate_holo_gated(self):
            """Function to calculate and project a series of point cloud CGH with
            different exposures.
            """
            # definition of intensities vector, here all points have same intensity
            phases=[]
            exposures=[]
            intensities=np.ones(self.coords_holo.shape[0])
            print(self.coords_holo.shape, self.coords_holo)
            #vector to list
            self.int_list=intensities.tolist()
            #Function to define the CGH
            holo_target=hologram(points = self.coords_holo,intensities=self.int_list,\
                        aberrations = None,playerInst = self.SLM,sourceIndex = 0,\
                        apertureIndex=0)
            #Function to calculate the CGH, here we use CSWGS as algo
            holo_target.compute("CSWGS",100,0.5)
            holo_target.wait()
            print('hologram ready!')
            #Save CGH as phase pattern in the above chosen directory
            np.save(self.path_now+'phase_'+self.name_holo.input.text(),holo_target.phase)
            # Load Cgh on SLM with exposure time
            phases.append(holo_target.phase)
            exposures.append(self.exp_time)
            print(exposures)
            print ("setting phase")
            self.SLM.setSequence(phases,exposures)
            self.SLM.play()
            # time.sleep(4.5)# sum of the exposures + error in s
            time.sleep(self.duration_seq)# sum of the exposures + error in s

            self.SLM.stop()
    #--------------------------FFT holo with convolution approach ------------------
    #Here we are implementing the convolution approach to generate an fft holo.
    #We calculate an imput  mask image as  a circle centered at coords 0,0 in the
    #sample with radius r. This mask will be added as isoplanatic aberration to a
    #point cloud CGH. The result will be an extended CGH centered at the point cloud
    #pattern.
    #-------------------------------------------------------------------------------
        def calculate_fft_holo_convolution(self):
            """Function to calculate fft CGH with convolution approach. A point cloud
            CGH of the seclected points is calculated and projected.
            This CGH become extended by loading on the SLM the phase of a
            circle of radius r as aberration.
            """

            # Calulation of point-cloud CGH for centroids
            intensities=np.ones(self.coords_holo.shape[0])
            self.int_list=intensities.tolist()
            holo_target=hologram(points = self.coords_holo,intensities=self.int_list,\
                        aberrations = None,playerInst = self.SLM,sourceIndex = 0,\
                        apertureIndex=0)
            holo_target.compute("CSWGS",100,0.5)
            holo_target.wait()
            #load on SLM point cloud CGH
            self.SLM.setHologram(holo_target)
            time.sleep(.5)
            np.save(self.path_now+'phase_point_fft'+self.name_holo.input.text(),holo_target.phase)
            #Load on SLM fft CGH of mask as isoplanatic aberation.
            self.SLM.setIsoAberrationPhase(self.holo_mask_phase)
            time.sleep(.5)
            print('fft hologram ready!')
    #-------------------------------------------------------------------------------
        def set_radius(self):
            """Function to insert the radius for the extended CGH
            """
            self.r  =float(self.r_input.input.text())
            print("Radius set")
    #-------------------------------------------------------------------------------
        def get_mask(self):
            """Function to calculate the input mask image for fft calculation
            """
            mask_dim = 400
            coords_mask = np.zeros((1,3))
            # mask definition
            mask=np.zeros((mask_dim ,mask_dim),dtype=float)
            x = np.arange(-mask_dim/2,mask_dim/2,1)
            y = np.arange(-mask_dim/2,mask_dim/2,1)
            x, y = np.meshgrid(x, -y)
            for j in range(coords_mask.shape[0]):
                mask[(x-(coords_mask[j,0]))**2 + \
                    (y-(coords_mask[j,1]))**2<= self.r**2] = 1
            # save mask image and z coords as global variables in list
            self.list_mask = []
            self.z_mask = []
            self.list_mask.append(mask)
            self.z_mask.append(coords_mask[:,2].tolist())
            print('mask list ready!')

            #Calculation of fft CGH of image mask
            pix_size= .5
            self.holo_mask=hologramFft(self.list_mask,self.z_mask[0],[pix_size],[True],self.SLM,0,0)
            self.holo_mask.compute("WGS",100)
            self.holo_mask.wait()
            #save fft CGH of image mask as phase pattern
            np.save(self.path_now+'fft_holo_mask_pix_.5_r_'+str(self.r),self.holo_mask.phase)
            self.holo_mask_phase=self.holo_mask.phase
            print('mask ready!')



    #-------------------------------------------------------------------------------
        def load_mask(self):
            """Function to load  the CGH of a mask as a npy file.
        
            """
            print('path now',self.path_now)
            temp= qtw.QFileDialog.getOpenFileName(None,"Select mask for fft holo",
                                            self.path_now)
            self.mask_path=temp[0]
            print('MASK selected ', self.mask_path)
            self.holo_mask_phase = np.load(self.mask_path)
     
    #-------------------------------------------------------------------------------
        def save_pixmap(self):
            # d=or cycle to change pixmap and save each one
            # for i in numnero immagini in volume
                #chiamoa update in select points
            # print(str(self.savedata_path)+\
            #                             "pixmap_z_"+str(self.select_points.stack_index)+'.tif')
            # self.select_points.pix.save(str(self.savedata_path)+self.holo_name+'_'+\
            #                             str(self.select_points.stack_index)+'.tif')
            self.select_points.pix.save(str(self.savedata_path)+str(self.volume_slider).zfill(6)+"index_pixmap_z_"+str(self.select_points.stack_index)+'.tif')                           
                #load with tiffffile
                #use numpy scipy to reascale
                #re-save


    #-------------------------------------------------------------------------------
        def metadata_timelapse(self):
            """Function to save metadata
            """


            if(str(self.metadata_timelapse_button.currentText())=='TL'):
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))], \
                ['laser source',self.s_lasers],\
                ['n_planes_timelapse',self.plane_scan.planes]],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_TL_'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()



            if(str(self.metadata_timelapse_button.currentText())=='TL+2P_single'):
                self.calculation_dt_2P_single()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['n_planes_timelapse',self.plane_scan.planes], ['laser source',self.s_lasers],\
                ['t_on_2P',self.shutter.t_on],['CGH_coords',self.coords_holo],\
                ['dt_im_2P_single', self.dt_2P_single]],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_TL+2P_single_'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()


                #Save in dictionaries timings
                self.values_timing_2P_single.append(self.shutter.t_2P_single)
                self.values_timing_2P_single.append(self.time_stamp_2P_single)

                self.values_timing_im.append(self.plane_scan.t0_im)
                self.values_timing_im.append(self.time_stamp_im)

                for i in range(len(self.field_timing)):
                    self.dict_timing_2P_single[self.field_timing[i]]=self.values_timing_2P_single[i]
                    self.dict_timing_im[self.field_timing[i]]=self.values_timing_im[i]

                pickle.dump(self.dict_timing_2P_single, open(self.savedata_path+'timing_2P_TL+2P_single'+timestr+".p", "wb"))
                pickle.dump(self.dict_timing_im, open(self.savedata_path+'timing_im_TL+2P_single'+timestr+".p", "wb"))
                pickle.dump(self.plane_scan.shutter_rec, open(self.savedata_path+'shutter_rec'+timestr+".p", "wb"))
                # np.save(self.savedata_path+'shutter_rec'+timestr, self.plane_scan.shutter_rec)

                #clear variables for next run
                # self.shutter.t_2P_single=[]
                self.shutter.clear_timing()
                self.plane_scan.clear_timing()
                self.time_stamp_2P_single=[]
                self.dict_timing_2P_single={}
                self.values_timing_im=[]
                self.values_timing_2P_single=[]


            if(str(self.metadata_timelapse_button.currentText())=='TL+2P_train'):
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['n_planes_timelapse',self.plane_scan.planes],['laser source',self.s_lasers],\
                ['t_on_2P',self.shutter.t_on], ['frequency_2P', self.shutter.frequency],\
                ['CGH_coords',self.coords_holo],\
                ['n_pulses_2P', self.shutter.n_pulses],['dt_im_2P_train',  self.dt_2P_train]],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_TL+2P_train'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()

                #Save in dictionaries timings
                self.values_timing_2P_train.append(self.shutter.t_2P_train)
                self.values_timing_2P_train.append(self.time_stamp_2P_train)


                self.values_timing_im.append(self.plane_scan.t0_im)
                self.values_timing_im.append(self.time_stamp_im)


                for i in range(len(self.field_timing)):
                    self.dict_timing_2P_train[self.field_timing[i]]=self.values_timing_2P_train[i]
                    self.dict_timing_im[self.field_timing[i]]=self.values_timing_im[i]

                pickle.dump(self.dict_timing_2P_train, open(self.savedata_path+'timing_2P_TL+2P_train'+timestr+".p", "wb"))
                pickle.dump(self.dict_timing_im, open(self.savedata_path+'timing_im_TL+2P_train'+timestr+".p", "wb"))
                pickle.dump(self.plane_scan.shutter_rec, open(self.savedata_path+'shutter_rec'+timestr+".p", "wb"))
                # np.save(self.savedata_path+'shutter_rec'+timestr, self.plane_scan.shutter_rec)
                #clear variables for next run
                self.shutter.clear_timing()
                self.dict_timing_2P_train={}
                self.time_stamp_2P_train=[]
                self.values_timing_im=[]
                self.values_timing_2P_train=[]
                self.shutter.clear_timing()


            if(str(self.metadata_timelapse_button.currentText())=='z+2P_single'):
                self.calculation_dt_2P_single()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['dz (um)',self.z_step_um ],\
                ['n_planes',self.volume_scan.planes],['laser source',self.s_lasers],\
                ['t_on_2P',self.shutter.t_on],\
                ['n_volumes',self.volume_scan.volumes],\
                ['CGH_coords',self.coords_holo],\
                ['dt_im_2P_single', self.dt_2P_single]],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_TL+2P_single_'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()


                #Save in dictionaries timings
                self.values_timing_2P_single.append(self.shutter.t_2P_single)
                self.values_timing_2P_single.append(self.time_stamp_2P_single)

                self.values_timing_im.append(self.volume_scan.t0_zstack)
                self.values_timing_im.append(self.time_stamp_im)
                print(self.time_stamp_im, self.volume_scan.t0_zstack)

                for i in range(len(self.field_timing)):
                    self.dict_timing_2P_single[self.field_timing[i]]=self.values_timing_2P_single[i]
                    self.dict_timing_im[self.field_timing[i]]=self.values_timing_im[i]

                pickle.dump(self.dict_timing_2P_single, open(self.savedata_path+'timing_2P_zstack+2P_single'+timestr+".p", "wb"))
                pickle.dump(self.dict_timing_im, open(self.savedata_path+'timing_im_zstack+2P_single'+timestr+".p", "wb"))
                pickle.dump( self.plane_scan.shutter_rec, open(self.savedata_path+'shutter_rec'+timestr+".p", "wb"))
                #clear variables for next run
                # self.shutter.t_2P_single=[]
                self.shutter.clear_timing()
                self.plane_scan.clear_timing()
                self.time_stamp_2P_single=[]
                self.dict_timing_2P_single={}
                self.values_timing_im=[]
                self.values_timing_2P_single=[]


            if(str(self.metadata_timelapse_button.currentText())=='z+2P_train'):
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['dz (um)',self.z_step_um ],\
                ['n_planes',self.volume_scan.planes], ['laser source',self.s_lasers],\
                ['n_volumes',self.volume_scan.volumes],\
                ['t_on_2P',self.shutter.t_on], \
                ['frequency_2P', self.shutter.frequency],['n_pulses_2P', self.shutter.n_pulses],\
                ['dt_im_2P_train',  self.dt_2P_train], ['CGH_coords',self.coords_holo]],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_z+2P_train'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()

                #Save in dictionaries timings
                self.values_timing_2P_train.append(self.shutter.t_2P_train)
                self.values_timing_2P_train.append(self.time_stamp_2P_train)


                self.values_timing_im.append(self.volume_scan.t0_zstack)
                self.values_timing_im.append(self.time_stamp_im)
                print(self.time_stamp_im, self.volume_scan.t0_zstack)


                for i in range(len(self.field_timing)):
                    self.dict_timing_2P_train[self.field_timing[i]]=self.values_timing_2P_train[i]
                    self.dict_timing_im[self.field_timing[i]]=self.values_timing_im[i]

                pickle.dump(self.dict_timing_2P_train, open(self.savedata_path+'timing_2P_zstack+2P_train'+timestr+".p", "wb"))
                pickle.dump(self.dict_timing_im, open(self.savedata_path+'timing_im_zstack+2P_train'+timestr+".p", "wb"))
                # pickle.dump(self.volume_scan_shutter.shutter_rec, open(self.savedata_path+'shutter_rec'+timestr+".p", "wb"))
                
                #clear variables for next run
                self.shutter.clear_timing()
                self.dict_timing_2P_train={}
                self.time_stamp_2P_train=[]
                self.values_timing_im=[]
                self.values_timing_2P_train=[]
                self.volume_scan.clear_timing_zstack()





            if(str(self.metadata_timelapse_button.currentText())=='TL_z_mc'):
                timestr = time.strftime("%Y%m%d-%H%M%S")
                table_results=tabulate([['Frequency im', self.acquisition_frequency],\
                ['scanning speed',self.image_interaction.line_scan_speed],\
                ['slit height',self.image_interaction.exposed_pixel_height],\
                ['z_pos',self.displace_z_value.input.text()],\
                ['z_min',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['dz (um)',self.z_step_um ],\
                ['n_planes',self.volume_scan.planes],\
                ['n_volumes',self.volume_scan.volumes],\
                # ['t_on_2P',self.shutter.t_on], \
                # ['frequency_2P', self.shutter.frequency],['n_pulses_2P', self.shutter.n_pulses],\
                # ['dt_im_2P_train',  self.dt_2P_train], ['CGH_coords',self.coords_holo]
                ],\
                headers=['Acquisition parameters', 'Values'])

                f = open(self.savedata_path+'metadata_TL_z_multicolor'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()

                #Save in dictionaries timings
                # self.values_timing_2P_train.append(self.shutter.t_2P_train)
                # self.values_timing_2P_train.append(self.time_stamp_2P_train)




                self.dict_timing_im_multic[self.field_timing[0]]=self.values_timing_im_green
                self.dict_timing_im_multic[self.field_timing[1]]=self.values_timing_im_blue

                # pickle.dump(self.dict_timing_2P_train, open(self.savedata_path+'timing_2P_zstack+2P_train'+timestr+".p", "wb"))
                pickle.dump(self.dict_timing_im_multic, open(self.savedata_path+'timing_im_multic'+timestr+".p", "wb"))
                # pickle.dump(self.volume_scan_shutter.shutter_rec, open(self.savedata_path+'shutter_rec'+timestr+".p", "wb"))
                
                #clear variables for next run
                # self.shutter.clear_timing()
                # self.dict_timing_2P_train={}
                # self.time_stamp_2P_train=[]
                # self.values_timing_im=[]
                # self.values_timing_2P_train=[]
                # self.volume_scan.clear_timing_zstack()
                self.values_timing_im_green=[]
                self.values_timing_im_green=[]



    #-------------------------------------------------------------------------------
        def metadata_zstack(self):
            """Function to save metadata
            """
            timestr = time.strftime("%Y%m%d-%H%M%S")
            if(str(self.metadata_zstack_button.currentText())=='TL_zstack'):
                table_results=tabulate([['Frequency im (Hz)', self.acquisition_frequency],\
                ['scanning speed (pixels/s)',self.image_interaction.line_scan_speed],\
                ['slit height (pixels)',self.image_interaction.exposed_pixel_height],\
                ['dz (um)',self.z_step_um ],\
                ['z_min (um)',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max (um)',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['n_planes',self.volume_scan.planes],\
                ['n_volumes',self.volume_scan.volumes],\
                ['laser source',self.s_lasers]],\
                headers=['Acquisition parameters', 'Values'])
            
                f = open(self.savedata_path+'metadata_TL_zstack'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()
        
            if(str(self.metadata_zstack_button.currentText())=='z_stack'):
                table_results=tabulate([['Frequency im (Hz)', self.acquisition_frequency],\
                ['scanning speed (pixels/s)',self.image_interaction.line_scan_speed],\
                ['slit height (pixels)',self.image_interaction.exposed_pixel_height],\
                ['dz (um)',self.z_step_um ],\
                ['z_min (um)',str(np.round(self.galvo_microns_per_volts_factor * self.z_min_set, 4))],\
                ['z_max (um)',str(np.round(self.galvo_microns_per_volts_factor * self.z_max_set, 4))],\
                ['n_planes',self.volume_scan.planes],\
                ['laser source',self.s_lasers]],\
               
                headers=['Acquisition parameters', 'Values'])
            
                f = open(self.savedata_path+'metadata_zstack'+timestr+'.txt', 'w')
                f.write(table_results)
                f.close()  

    #-------------------------------------------------------------------------------
        def metadata_cgh(self):
            """Function to save metadata
            """
            timestr = time.strftime("%Y%m%d-%H%M%S")
            table_results=tabulate([['selected coords',self.select_points.coords],\
            ['CGH coords',self.coords_holo],['radius cgh',self.r]], \
            headers=['CGH parameters', 'Values'])

            f = open(self.savedata_path+'metadata_cgh'+timestr+'.txt', 'w')
            f.write(table_results)
            f.close()

    #-------------------------------------------------------------------------------
    # TO WORK ON CANOPY
    #-------------------------------------------------------------------------------
    # To work on Canopy + style
    app = qtw.QApplication.instance()
    stylesheet = \
        'C:\\style_gui\\gui_style.qss'
    standalone = app is None
    if standalone:
        app = qtw.QApplication(sys.argv)
    with open(stylesheet,"r") as style:
        app.setStyleSheet(style.read())
    execute = neurinvestigation()
    print('welcome')
    if standalone:
        sys.exit(app.exec_())
    else:
        print("\n *** Program started *** \n")
