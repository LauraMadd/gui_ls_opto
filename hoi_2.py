from importlib import reload
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import numpy as np
# from hands_on_image import hands_on as hoi
import hands_on_image
reload(hands_on_image)
hoi=hands_on_image.HandsOn
from skimage.transform import resize
# from try_mm_py36 import camera
from MM_camera_GUI import camera
# reload(camera)
import time

class acquisition_thread(qtc.QObject):
    """ This class is basically self explanatory.
     Every method clear the buffer and prepare the camera for the acquisition trigger.
     It inherits from QObject only because we want to put it in a thread,
     the QObject have a super easy function to do that directly from the GUI program.
    """
    
    signalStatus = qtc.pyqtSignal(str)

    def __init__(self, image,  camera_params = \
                ['ThorCam','ThorlabsUSBCamera','ThorCam'], image_size=1024, \
                                                                parent=None):
        """
        Camea is a cam object
        Image is a hoi object! Pay attention!
        It is a class containing two old classes.
        Just because we need to add the QObject properties.
        """
        super(self.__class__, self).__init__(parent)
        self.params = camera_params
        self.image = image
        self.cam = camera(camera_params)
        self.cam.mmc.setCircularBufferMemoryFootprint(1000)
        self.cam.mmc.setProperty(self.cam.name, 'TriggerMode', 'External')
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-ScanSpeedControlEnable', 'On')       
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-SensorReadoutMode', \
                                'Bottom Up Sequential') 
        self.cam.mmc.setProperty(self.cam.name,
                                'LightScanPlus-AlternatingReadoutDirection',
                                'On')   
        self.cam.mmc.setProperty(self.cam.name,
                                'Sensitivity/DynamicRange',
                                '16-bit (low noise & high well capacity)') 
        self.cam.savedata_path='C:\\Laura\\test_gui2\\'
        self.stop_signal = False
        self.image_size = image_size
        
        self.line_scan_speed = 25000
        self.exposed_pixel_height = 200
        
        self.acquired_volume = []
        self.acquired_timelapse=[]
        self.acquired_volume_resized = []
        self.images_per_volume = 1
        self.acquisition_frequency = 1
        
        self.volume_done = False
        
        self.acquisition_on = False
        
    @qtc.pyqtSlot()        
    def start_live(self):
        # default initialization
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-LineScanSpeed [lines/sec]', \
                                str(self.line_scan_speed))                      
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-ExposedPixelHeight', \
                                str(self.exposed_pixel_height))
        self.cam.mmc.clearCircularBuffer()
        self.cam.mmc.prepareSequenceAcquisition(self.cam.name)
        self.cam.mmc.startContinuousSequenceAcquisition(1)
        print('\n Camera eyes open (O_O)')
        image_count = 0
        # images are updated so that they can be displayed in "image"
        while(self.stop_signal != True): 
            while (self.cam.mmc.getBufferTotalCapacity()==\
                            self.cam.mmc.getBufferFreeCapacity()):
                                pass
            ##LM flip lr
            self.resized = resize(np.fliplr(self.cam.mmc.popNextImage()),\
                        (self.image_size, self.image_size), anti_aliasing=True)

            self.image.update_image(self.resized)
            image_count +=1
        self.cam.mmc.stopSequenceAcquisition()
        self.stop_signal = False
        print(image_count, ' images transmitted\n')
    
    @qtc.pyqtSlot() 
    def start_single_volume_acquisition(self):
        self.volume_done = False
        self.acquired_volume, self.acquired_volume_resized = [], []
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-LineScanSpeed [lines/sec]', \
                                str(self.line_scan_speed))                      
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-ExposedPixelHeight', \
                                str(self.exposed_pixel_height))
        self.cam.mmc.clearCircularBuffer()
        self.cam.mmc.prepareSequenceAcquisition(self.cam.name)
        self.cam.mmc.startSequenceAcquisition(self.images_per_volume,
                                                0,
                                                False)
        print('Camera eyes open O_O')
        
        time.sleep(.2)
        
        while(self.cam.mmc.getBufferTotalCapacity() \
                - self.cam.mmc.getBufferFreeCapacity() < self.images_per_volume):
                pass
        while (self.cam.mmc.getBufferTotalCapacity()!= \
                                        self.cam.mmc.getBufferFreeCapacity()):
            temp_image =self.cam.mmc.popNextImage()
            ##LM flip lr
            temp_image=np.fliplr(temp_image)
            self.acquired_volume.append(temp_image)
            self.resized = resize(temp_image,\
                        (self.image_size, self.image_size), anti_aliasing=True)
            self.acquired_volume_resized.append(self.resized)
        self.cam.mmc.stopSequenceAcquisition()
        print('single volume acquired (-.-)')
        self.volume_done = True
        
    @qtc.pyqtSlot()
    def TL_continuos_saving_acquisition(self):
        self.cam.saving_TL_matrices_as_tiff()


    @qtc.pyqtSlot()
    def zstack_continuos_saving_acquisition(self):
        self.cam.saving_z_stack_matrices_as_tiff()


    @qtc.pyqtSlot()
    def TL_zstack_continuos_saving_acquisition(self):
        self.cam.saving_TL_z_stack_matrices_as_tiff()


    @qtc.pyqtSlot()        
    def start_live_save(self):
        # default initialization
        self.acquired_timelapse=[]
   
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-LineScanSpeed [lines/sec]', \
                                str(self.line_scan_speed))                      
        self.cam.mmc.setProperty(
                                self.cam.name,\
                                'LightScanPlus-ExposedPixelHeight', \
                                str(self.exposed_pixel_height))
        self.cam.mmc.clearCircularBuffer()
        self.cam.mmc.prepareSequenceAcquisition(self.cam.name)
        self.cam.mmc.startContinuousSequenceAcquisition(1)
        print('\n Camera eyes open (O_O)')
        image_count = 0
        self.acquired_timelapse=[]
   
        # images are updated so that they can be displayed in "image"
        while(self.stop_signal != True): 
            while (self.cam.mmc.getBufferTotalCapacity()==\
                            self.cam.mmc.getBufferFreeCapacity()):
                                pass
            ##LM flip lr
            temp_im=np.fliplr(self.cam.mmc.popNextImage())
            #prepare  to save LM 
            self.acquired_timelapse.append(temp_im)
            self.resized = resize(temp_im,\
                        (self.image_size, self.image_size), anti_aliasing=True)

            self.image.update_image(self.resized)
            image_count +=1
            # print(image_count, ' images transmitted\n')
        self.cam.mmc.stopSequenceAcquisition()
        self.stop_signal = False
        print('Done!',image_count, ' images transmitted\n')
        self.cam.mmc.clearCircularBuffer()







        

# example
# ciao = acquisition_thread((hoi(np.random. rand(500,500)), \
#                             ['ThorCam','ThorlabsUSBCamera','ThorCam']))
        
        
        