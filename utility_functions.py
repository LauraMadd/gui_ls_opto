import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import fnmatch
import math_functions as mtm
# import skimage.external.tifffile as scitif
from scipy.ndimage import median_filter
from PIL import Image
from time import sleep
# import hands_on_image as hoi
import pickle
from PyQt5 import QtGui, QtCore
import os
import nudged
# from SLMcontroller import hologram

#
#

def set_ROI_planes (ETL, SLM, semi_path, current_value,volume_flag= True,index_pattern=0):
    """ Sets a current value for the lens and a pattern on the SLM. 
        This function is mainly used in a for loop over the currents and a list
        of patterns with different z coordinate. If the volume flag is False it 
        works also for a single pattern in one plane but in this case the user
        needs to fill in  the pattern name in the semi_path variable 
        
        Inputs: 
        -ETL, SLM (works with SLM_controller Pozzi)
        -sempi_path= path containing the holograms 
        -index= identifies the pattern and the associated focal lenght of the ETL 
        -current_value= current of ETL corresponding to the index selected 
    """
    
    ETL.current(current_value)
    if(volume_flag==True):
        patterns=fnmatch.filter(ETL.listdir(semi_path), '*.npy')
        name_holo=patterns[index_pattern] 
        holo=np.load(semi_path+name_holo)
    else:
        holo=np.load(semi_path)
    print ("setting phase")
    SLM.setPhase(holo) 
    sleep(.4)

    return None 
#-------------------------------------------------------------------------------   
def set_ROI_3D (ETL, SLM, semi_path, current_value):
    """ Sets a current value for the lens and a pattern on the SLM. 
        Inputs: 
        -ETL, SLM (works with SLM_controller Pozzi)
        -sempi_path= path containing the hologram
        -current_value= current of ETL corresponding to thr hologram
         
    """
    ETL.current(current_value)
   
    holo=np.load(semi_path)
    print ("setting phase")
    SLM.setPhase(holo) 

    return None
#------------------------------------------------------------------------------- 




# def Stack_registration (stack_path,save_path,n_frames=None):
#     """ Realigns the images in a stack and save them as single tif images 
#         Inputs:
#         -location path of the stack+filename
#         -location path of the realigned images with \\
#         -flag to apply median filter on each frame, number of frames in the image 
#         !!! TO BE TESTED
#     """
#     if(n_frames != None):
#         images=[]
#         for i in range(n_frames):
#             stack_path.seek(i)
#             images.append(median_filter(np.asarray(stack_path),3))
#         stack=np.asarray(images)
#         
#     else:
#         stack=scitif.imread(stack_path)
#    
#     mtm.realignStack(stack)  
#     for i in range (stack.shape[0]):
#         Image.fromarray(stack[i,:,:]).save(save_path+str(i).zfill(2)+".tif")
#     
#     return None 
#------------------------------------------------------------------------------- 
# def holo_coords(image, show = True, snap_label = 'default'):
#        
#     iter_roi_y0=0
#     iter_roi_x0=0
#     im_matrix = np.array(image)
#     if(show):
#         # careful: from 16 to 8bit conversion just to use PyQt
#         # the actual image remain 16bit
#         if (im_matrix.shape[0] > 1000 or im_matrix.shape[1] > 1000):
#             print\
#             ( '\nmore than 1000 pixels:preview downsized to be seen comfortably.\n')
#             converted = mtm.rebin(im_matrix, [int(im_matrix.shape[0]/2),\
#                                 int(im_matrix.shape[1]/2)])
#             converted_flag = True
#             converted_8 =(converted*255/np.amax(converted)).astype(np.uint8)
#                 # converted  =((converted - converted.min()) /\
#                 #     (converted.ptp() / 65536.0)).astype(np.uint16)
#             result = QtGui.QImage(converted_8.data, converted_8.shape[1], \
#                        converted_8.shape[0], QtGui.QImage.Format_Indexed8)
#                 # call classe hands_on_image, basically a modified PyQt label,
#                 # to hold an the image and select a roi.
#             faino = hoi.hands_on_image(result)
#             faino.resize(converted.shape[1],converted.shape[0])
#             faino.show()
#         else:
#             converted = im_matrix.astype(np.uint8)
#             # converted  =((self.snapped - self.snapped.min()) /\
#             #     (self.snapped.ptp() / 65536.0)).astype(np.uint16)
#             result = QtGui.QImage(converted.data, converted.shape[1], \
#                         converted.shape[0], QtGui.QImage.Format_Indexed8)
#             faino = hoi.hands_on_image(result)
#             faino.resize(converted.shape[1],converted.shape[0])
#             faino.show()
#             converted_flag = False
#                 
#     if (converted_flag == False):
#         coord_y = iter_roi_x0 + faino.xStart
#         coord_x = iter_roi_y0 + faino.yStart
# 
#         print ( coord_x,coord_y)
#            
#           
#     else:
#         coord_y = iter_roi_x0 +faino.xStart * 2
#         coord_x = iter_roi_y0 + faino.yStart * 2
#             
#         print ( coord_x,coord_y)
#             
#         
#     return   coord_x,coord_y       
#-------------------------------------------------------------------------------
    
def getCentroids(img_array, thresh, BOX_WIDTH, BOX_HEIGHT,n_points):
    """ Calculates the centroids from an input image after doing the thresholding
        of the bg noise on the image , this function works for rectangular grids 
         in fact there are 2 box widths. 
         This function is used for affine T 
    """
    # img_array = threshold(img_array, thresh)
    img_array[img_array[:,:] <= thresh] = 0
    temp = np.ones((img_array.shape))

    centers=np.zeros((n_points,2))
    # Algorithm
    for i in range(n_points):
        max_index = np.unravel_index(img_array.argmax(), img_array.shape)
        # print(max_index)
        centers[i,0] = (max_index[0])
        centers[i,1] = (max_index[1])
        img_array[max_index[0]-BOX_WIDTH: max_index[0]+BOX_WIDTH,
                 max_index[1]-BOX_HEIGHT: max_index[1]+BOX_HEIGHT] = 0
        temp[max_index[0]-BOX_WIDTH: max_index[0]+BOX_WIDTH,
                 max_index[1]-BOX_HEIGHT: max_index[1]+BOX_HEIGHT] = 0
    # plt.figure('point'+str(i))
    # plt.imshow(temp, interpolation = 'none')
    # plt.show()

    return centers
#-------------------------------------------------------------------------------
def sort_camera_coord(input_coord, ref_coord):
    """ Function to sort coordinated found with getCentroids on a camera image
        Input:
            -input_ccord= coord to sort
            -ref coord= coord set as reference, given as numpy arrays , they are
            the format used to generate digital holograms
        This function is used in T_affine and works for setup version december19 
    """
    distance = np.zeros((input_coord.shape[0]))
    ordered_coord = np.zeros((input_coord.shape))
    minimal_dist=np.zeros((input_coord.shape),dtype=np.uint8)
    for i in range(input_coord.shape[0]):
        
        for j in range(input_coord.shape[0]):
            distance[j] = np.sqrt((input_coord[i,0]-ref_coord[j,0])**2+(input_coord[i,1]-ref_coord[j,1])**2)
        # print(distance)
        # print('distance'), print(distance)
        minimal_dist[i] = np.argmin(distance)
        ordered_coord[minimal_dist[i], 0] = input_coord[i, 0] 
        ordered_coord[minimal_dist[i], 1] = input_coord[i, 1] 
   
    # print('index') , print(minimal_dist)
    return ordered_coord
#-------------------------------------------------------------------------------
def Find_Taffine(coord_ref,save_path,img_name,bg_name,num_points, box_width,box_height,thresh_max,thresh_min):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;   
    
    
    #Moving point from camera space
    centers=np.zeros((num_points,2))
    
    im1=Image.open(img_name)
    img_original = np.array(im1).astype(np.uint16) 
       
  
    im2=Image.open(bg_name)
    img_bg = np.array(im2).astype(np.uint16)
        
        
        
    img =np.zeros((cam_h,cam_h), dtype=np.uint16)
    img=img_original-img_bg
    plt.figure('No bg image')
    plt.imshow(img)
        
    img_ht=mtm.kill_hot_pixels_bis(img,thresh_max)
    plt.figure('read centroids control')
    plt.imshow(img_ht, interpolation = 'none')
    plt.show()
   
        
    centers=getCentroids(img_ht, thresh_min, int(box_width/2), int(box_height/2), num_points)


    plt.figure('control bis')
    plt.scatter(centers[:,1], centers[:,0])
    plt.show()
    


    #calculation centroids position in a cartesian system of reference
    xc_im=centers[:,0]-off_x;
    yc_im=centers[:,1]-off_y
    
    #sample plane
    centers_um=np.zeros((num_points,2), dtype=np.float64)
    centers_um[:,0]=xc_im*M_2*pix_size
    centers_um[:,1]=yc_im*M_2*pix_size
    
    coord_moving=sort_camera_coord(centers_um,coord_ref)
    np.save(save_path+'coord_moving',coord_moving)
        
    #T affine calculation
    T_affine=nudged.estimate(coord_moving, coord_ref)
    # T affine matrix
    T_matrix=T_affine.get_matrix()
    np.save(save_path+'T_maxtrix',T_matrix)

    
    return T_affine,coord_moving
    
#-------------------------------------------------------------------------------
def Find_Taffine_live(coord_ref,save_path,img_raw,num_points, box_width,box_height,thresh_max,thresh_min):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;   
    
    
    #Moving point from camera space
    centers=np.zeros((num_points,2))
    
  #   im1=Image.open(img_name)
  #   img_original = np.array(im1).astype(np.uint16) 
  #      
  # 
  #   im2=Image.open(bg_name)
  #   img_bg = np.array(im2).astype(np.uint16)
        
        
        
    # img =np.zeros((cam_h,cam_h), dtype=np.uint16)
    # img=img_raw
    # plt.figure('No bg image')
    # plt.imshow(img)
        
    img_ht=mtm.kill_hot_pixels_bis(img_raw,thresh_max)
    # plt.figure('read centroids control')
    # plt.imshow(img_ht, interpolation = 'none')
    # plt.show()
   
        
    centers=getCentroids(img_ht, thresh_min, int(box_width/2), int(box_height/2), num_points)

# 
    # plt.figure('control bis')
    # plt.scatter(centers[:,1], centers[:,0])
    # plt.show()
    


    #calculation centroids position in a cartesian system of reference
    xc_im=centers[:,0]-off_x;
    yc_im=centers[:,1]-off_y
    
    #sample plane
    centers_um=np.zeros((num_points,2), dtype=np.float64)
    centers_um[:,0]=xc_im*M_2*pix_size
    centers_um[:,1]=yc_im*M_2*pix_size
    
    coord_moving=sort_camera_coord(centers_um,coord_ref)
    # np.save(save_path+'coord_moving_Taffine',coord_moving)
    
    # print('micron'),print(centers_um)
    # print('micron ordered'), print(coord_moving)    
    #T affine calculation
    T_affine=nudged.estimate(coord_moving, coord_ref)
    
    # T affine matrix
    # T_matrix=T_affine.get_matrix()
    # np.save(save_path+'T_maxtrix',T_matrix)
    
    # 
    # list_ref=coord_ref.tolist()
    list_moving=coord_moving.tolist()
    list_transf=T_affine.transform(list_moving[:])
    coord_transf=np.asarray(list_transf)
    
    # print('ref'), print(coord_ref)
    # print('transf'), print(coord_transf)

    
    return T_affine, coord_moving
    
    
    
#-------------------------------------------------------------------------------
def estimation_Taffine_accuracy_live(coord_ref,coord_moving,T_affine,num_points,save_path):    
    
    
    
  
   
    list_moving=coord_moving.tolist()
    
    # mse=0
    # list_ref=coord_ref.tolist()
    # mse = nudged.estimate_error(T_affine,list_moving, list_ref)

    # transformation of points
    list_transf=T_affine.transform(list_moving[:])
    coord_transf=np.asarray(list_transf)
    

   
    
    dst=np.zeros((num_points),dtype=np.float64)
    dst =np.sqrt((coord_ref[:,0]-coord_transf[:,0])**2+(coord_ref[:,1]-coord_transf[:,1])**2) 
   
    np.save(save_path+'acc_calibration', dst)
    print('Average accuracy affine transformation:',np.average(dst), 'um')
    print('Upper limit accuracy affine transformation:',np.max(dst), 'um')
    # plt.ion()
    # fig=plt.figure('Accuracy affine transformation')
   #  plt.plot(dst, 'ob')
   #  plt.xlabel('Points')
   #  plt.ylabel(r'$ Distance \quad ( \mu m)$')
   #  plt.xticks(np.arange(1,num_points+1,1), ['1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
   # # 
   # #  # plt.ylim((0,1))
   # #  plt.show()
   #  plt.savefig(save_path+'distances.png', bbox_inches='tight')
     
    # fig.savefig(save_path+'distances.png', bbox_inches='tight')
    # plt.close(fig)
   
   

    return None
#-------------------------------------------------------------------------------
def Find_Taffine_3D_debug(coord_ref,save_path,matrix_im,num_points,n_planes, box_width,box_height,thresh_max,thresh_min):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;
  
          
    T_matrix=np.zeros((3,3,n_planes),float)
    # centers=np.zeros((num_points,2,n_planes))    #Moving point from camera space
    coord_moving=np.zeros((num_points,2,n_planes),dtype=float)
    coord_transf=np.zeros((num_points,2,n_planes),dtype=float)
    T_affine=[]
    dst=np.zeros((num_points,n_planes),dtype=np.float64)
    for i in range(n_planes): 
    
        centers=np.zeros((num_points,2))    #Moving point from camera space
    
    
  #   im1=Image.open(img_name)
  #   img_original = np.array(im1).astype(np.uint16) 
  #      
  # 
  #   im2=Image.open(bg_name)
  #   img_bg = np.array(im2).astype(np.uint16)
        
        
        
    # img =np.zeros((cam_h,cam_h), dtype=np.uint16)
    # img=img_raw
    # plt.figure('No bg image')
    # plt.imshow(img)
        
   #      img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
   #      plt.figure('read centroids control'+str(i))
   #      plt.imshow(img_ht, interpolation = 'none')
   #      plt.show()
   # 
   #      
   #      centers=getCentroids(img_ht, thresh_min, int(box_width/2), int(box_height/2), num_points)
         
        # img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
        # plt.figure('read centroids control'+str(i))
        # plt.imshow(img_ht, interpolation = 'none')
        # plt.show()
   
         
        centers=getCentroids(matrix_im[:,:,i], thresh_min, int(box_width/2), int(box_height/2), num_points)






        print(i,centers)
        plt.figure('control bis'+str(i))
        plt.scatter(centers[:,1], centers[:,0])
        plt.show()
    


    #calculation centroids position in a cartesian system of reference
        xc_im=centers[:,0]-off_x;
        yc_im=centers[:,1]-off_y
    
    #sample plane
        centers_um=np.zeros((num_points,2), dtype=np.float64)
        centers_um[:,0]=xc_im*M_2*pix_size
        centers_um[:,1]=yc_im*M_2*pix_size
    
        coord_moving[:,:,i]=sort_camera_coord(centers_um,coord_ref)
    # np.save(save_path+'coord_moving_Taffine',coord_moving)
    
    # print('micron'),print(centers_um)
        print('micron ordered plane'+str(i)), print(coord_moving[:,:,i])    
    #T affine calculation
        T=nudged.estimate(coord_moving[:,:,i], coord_ref)
        T_affine.append(T)
        # T affine matrix
        T_matrix[:,:,i]=T_affine[i].get_matrix()
        np.save(save_path+'T_maxtrix_'+str(i),T_matrix[:,:,i])
    
    # 
    # list_ref=coord_ref.tolist()
        list_moving=coord_moving[:,:,i].tolist()
        print(i,list_moving )
        list_transf=T_affine[i].transform(list_moving[:])
        print(i,list_transf )
        coord_transf[:,:,i]=np.asarray(list_transf)
        print(i,coord_transf[:,:,i] )
        dst=np.zeros((num_points,n_planes),dtype=np.float64)
        dst[:,i] =np.sqrt((coord_ref[:,0]-coord_transf[:,0,i])**2+(coord_ref[:,1]-coord_transf[:,1,i])**2) 
   
    # np.save(save_path+'acc_calibration', dst)
        print('Average accuracy affine transformation plane: ',str(i),np.average(dst), 'um')
        print('Upper limit accuracy affine transformation plane:',str(i),np.max(dst), 'um')
    
    # print('ref'), print(coord_ref)
    # print('transf'), print(coord_transf)

    
    return T_affine, T_matrix, coord_moving, coord_transf
#-------------------------------------------------------------------------------
def Find_Taffine_3D_debug_bis(coord_ref,save_path,matrix_im,num_points,n_planes, box_width,box_height,thresh_max,thresh_min):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;
  
    z=np.zeros((2)) 
    z[0]=17
      
    T_matrix=np.zeros((3,3,n_planes),float)
    # centers=np.zeros((num_points,2,n_planes))    #Moving point from camera space
    coord_moving=np.zeros((n_planes,num_points,3),dtype=float)
    coord_transf=np.zeros((n_planes,num_points,3),dtype=float)
    T_affine=[]
    
    dst=np.zeros((num_points,n_planes),dtype=np.float64)
    #Moving point from camera space
    centers=np.zeros((num_points,2))
    for i in range(n_planes):
    
  #   im1=Image.open(img_name)
  #   img_original = np.array(im1).astype(np.uint16) 
  #      
  # 
  #   im2=Image.open(bg_name)
  #   img_bg = np.array(im2).astype(np.uint16)
        
        
        
    # img =np.zeros((cam_h,cam_h), dtype=np.uint16)
    # img=img_raw
    # plt.figure('No bg image')
    # plt.imshow(img)
        
   #      img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
   #      plt.figure('read centroids control'+str(i))
   #      plt.imshow(img_ht, interpolation = 'none')
   #      plt.show()
   # 
   #      
   #      centers=getCentroids(img_ht, thresh_min, int(box_width/2), int(box_height/2), num_points)
         
        # img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
        # plt.figure('read centroids control'+str(i))
        # plt.imshow(img_ht, interpolation = 'none')
        # plt.show()
   
        centers=np.zeros((num_points,2))
        centers=getCentroids(matrix_im[:,:,i], thresh_min, int(box_width/2), int(box_height/2), num_points)
       





        print(i,centers)
        plt.figure('control bis'+str(i))
        plt.scatter(centers[:,1], centers[:,0])
        plt.show()
    


    #calculation centroids position in a cartesian system of reference
        xc_im=centers[:,0]-off_x;
        yc_im=centers[:,1]-off_y
    
    #sample plane
        centers_um=np.zeros((num_points,2), dtype=np.float64)
        centers_um[:,0]=xc_im*M_2*pix_size
        centers_um[:,1]=yc_im*M_2*pix_size
    
        coord_moving[i,:,:2]=sort_camera_coord(centers_um,coord_ref[0,:,:2])
        coord_moving[i,:,2]=z[i]
     
    # np.save(save_path+'coord_moving_Taffine',coord_moving)
    
    # print('micron'),print(centers_um)
        print('micron ordered plane'+str(i)), print(coord_moving[i,:,:])    
    #T affine calculation
        T=nudged.estimate(coord_moving[i,:,:], coord_ref[i,:,:])
        T_affine.append(T)
        # T affine matrix
        T_matrix[:,:,i]=T_affine[i].get_matrix()
        np.save(save_path+'T_maxtrix_'+str(i),T_matrix[:,:,i])
    
    # 
    # list_ref=coord_ref.tolist()
        list_moving=coord_moving[i,:,:].tolist()
        print(i,list_moving )
        list_transf=T_affine[i].transform(list_moving[:])
        print(i,list_transf)
        coord_transf[i,:,:]=np.asarray(list_transf)
        print(i,coord_transf[i,:,:] )
        dst=np.zeros((n_planes,num_points),dtype=np.float64)
        dst[:,i] =np.sqrt((coord_ref[i,:,0]-coord_transf[i,:,0])**2+(coord_ref[i,:,1]-coord_transf[i,:,1])**2+(coord_ref[i,:,2]-coord_transf[i,:,2])**2) 
   
    # np.save(save_path+'acc_calibration', dst)
        print('Average accuracy affine transformation plane: ',str(i),np.average(dst), 'um')
        print('Upper limit accuracy affine transformation plane:',str(i),np.max(dst), 'um')
    
    # print('ref'), print(coord_ref)
    # print('transf'), print(coord_transf)

    
    return T_affine, T_matrix, coord_moving, coord_transf
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def Find_Taffine_3D(coord_ref,save_path,matrix_im,num_points,n_planes, box_width,box_height,thresh_max,thresh_min):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;
    T_affine=[]
  
          
   
    # centers=np.zeros((num_points,2,n_planes))    #Moving point from camera space
    coord_moving=np.zeros((num_points,2,n_planes),dtype=float)
    coord_transf=np.zeros((num_points,2,n_planes),dtype=float)
    
    dst=np.zeros((num_points,n_planes),dtype=np.float64)
    for i in range(n_planes): 
    
        centers=np.zeros((num_points,2))    #Moving point from camera space
    
    
  #   im1=Image.open(img_name)
  #   img_original = np.array(im1).astype(np.uint16) 
  #      
  # 
  #   im2=Image.open(bg_name)
  #   img_bg = np.array(im2).astype(np.uint16)
        
        
        
    # img =np.zeros((cam_h,cam_h), dtype=np.uint16)
    # img=img_raw
    # plt.figure('No bg image')
    # plt.imshow(img)
        
   #      img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
   #      plt.figure('read centroids control'+str(i))
   #      plt.imshow(img_ht, interpolation = 'none')
   #      plt.show()
   # 
   #      
   #      centers=getCentroids(img_ht, thresh_min, int(box_width/2), int(box_height/2), num_points)
         
        # img_ht=mtm.kill_hot_pixels_bis(matrix_im[:,:,i],thresh_max)
        # plt.figure('read centroids control'+str(i))
        # plt.imshow(img_ht, interpolation = 'none')
        # plt.show()
   
         
        centers=getCentroids(matrix_im[:,:,i], thresh_min, int(box_width/2), int(box_height/2), num_points)




# 
# 
#         print(i,centers)
#         plt.figure('control bis'+str(i))
#         plt.scatter(centers[:,1], centers[:,0])
#         plt.show()
#     


    #calculation centroids position in a cartesian system of reference
        xc_im=centers[:,0]-off_x;
        yc_im=centers[:,1]-off_y
    
    #sample plane
        centers_um=np.zeros((num_points,2), dtype=np.float64)
        centers_um[:,0]=xc_im*M_2*pix_size
        centers_um[:,1]=yc_im*M_2*pix_size
    
        coord_moving[:,:,i]=sort_camera_coord(centers_um,coord_ref)
    # np.save(save_path+'coord_moving_Taffine',coord_moving)
    
    # print('micron'),print(centers_um)
        print('micron ordered plane'+str(i)), print(coord_moving[:,:,i])    
    #T affine calculation
        T=nudged.estimate(coord_moving[:,:,i], coord_ref)
        T_affine.append(T)

    
    # 
    # list_ref=coord_ref.tolist()
        list_moving=coord_moving[:,:,i].tolist()
        list_transf=T_affine[i].transform(list_moving[:])
        coord_transf[:,:,i]=np.asarray(list_transf)
        
        dst=np.zeros((num_points,n_planes),dtype=np.float64)
        dst[:,i] =np.sqrt((coord_ref[:,0]-coord_transf[:,0,i])**2+(coord_ref[:,1]-coord_transf[:,1,i])**2) 
   
    # np.save(save_path+'acc_calibration', dst)
        print('Average accuracy affine transformation plane: ',str(i),np.average(dst), 'um')
        print('Upper limit accuracy affine transformation plane:',str(i),np.max(dst), 'um')
    
    # print('ref'), print(coord_ref)
    # print('transf'), print(coord_transf)

    
    return T_affine
#-------------------------------------------------------------------------------
def estimation_Taffine_accuracy(coord_ref,coord_moving,T_affine,num_points,save_path):    
    
    
    
    #T affine accuracy
    list_ref=coord_ref.tolist()
    list_moving=coord_moving.tolist()
    mse=0
    # mse = nudged.estimate_error(T_affine,list_moving, list_ref)

    # transformation of points
    list_transf=T_affine.transform(list_moving[:])
    
    coord_transf=np.asarray(list_transf)
    acc=np.zeros((num_points,2))
    acc_x=coord_ref[:,0]-coord_transf[:,0]
    acc_y=coord_ref[:,1]-coord_transf[:,1]
    acc[:,0]=acc_x
    acc[:,1]=acc_y
   
    
    dst=np.zeros((num_points),dtype=np.float64)
    dst =np.sqrt((coord_ref[:,0]-coord_transf[:,0])**2+(coord_ref[:,1]-coord_transf[:,1])**2) 
    # dst = distance.cdist(coord_transf, coord_ref,'euclidean')
    np.save(save_path+'acc_calibration', dst)
    
    fig=plt.figure('Accuracy affine transformation')
    plt.plot(dst, 'ob')
    plt.xlabel('Points')
    plt.ylabel(r'$ Distance \quad ( \mu m)$')
    plt.xticks(np.arange(1,num_points+1,1), ['1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
   
    # plt.ylim((0,1))
    plt.show()
    fig.savefig(save_path+'distances.png', bbox_inches='tight')
    
    return None
    # return dst,mse,coord_transf
    
def Find_Taffine_z(coord_ref,matrix_im,n_planes):
    
    
    #System parameters
    cam_h=2048;
    off_y=cam_h/2
    off_x=cam_h/2
    pix_size=6.5;
    M_2=9./300;

  
          
   
    centers=np.zeros((3,n_planes))
    coord_moving=np.ones((4,n_planes), dtype=np.float64)
    # centers_um=np.copy(centers)
    for i in range(n_planes):
    
        max_index = np.unravel_index(matrix_im[i,:,:].argmax(), matrix_im[i,:,:].shape)
        print(max_index)
        centers[0,i] = (max_index[0])
        centers[1,i] = (max_index[1])
        
        xc_im=centers[0,i]-off_x;
        yc_im=centers[1,i]-off_y
    
        #sample plane
        
        coord_moving[0,i]=xc_im*M_2*pix_size
        coord_moving[1,i]=yc_im*M_2*pix_size
        # coord_moving[2,i]=....
        
    camera_to_slm_matrix=np.dot(coord_ref,np.linalg.pinv(coord_moving))
    
    #add ones to coords to be transformed and calculate dot product with matrix
    # example_slm_coordinates=numpy.array([x,y,z,1])
    # example_camera_coordinates=numpy.dot(slm_to_camera_matrix,coord_moving)
    coord_transf=np.zeros((3,n_planes))
    coord_transf=np.dot(camera_to_slm_matrix,coord_moving)
    
    dst=np.zeros((n_planes),dtype=np.float64)
    dst[:] =np.sqrt((coord_ref[0,:]-coord_transf[0,:])**2+(coord_ref[1,:]-\
                coord_transf[1,:])**2,(coord_ref[2,:]-coord_transf[2,:])**2) 
   
    # np.save(save_path+'acc_calibration', dst)
    print('Average accuracy affine transformation plane: ',np.average(dst), 'um')
    print('Upper limit accuracy affine transformation plane:',np.max(dst), 'um')
    


    
    return camera_to_slm_matrix
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def find_centers_roi (n_centers,radius,image,quadrants,showFlag=False):
   
    """ Finds the centroid of point features in an image  
        Inputs: 
        -n_centers= number of features to detect
        -radius=width of the box set to zero once the feature has been found
        -image=image with point-like features
        -quadrants=quadrands defined on the image to guide the detection of the features 
         This function is used for aniso aberration corrections 
    """
     # image = np.zeros((camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype = np.uint16)
     # image=camera.snap(True)
    image_2=mtm.kill_hot_pixels(image, 2**16,file_name = None )
    centers_coord=np.zeros((n_centers,2))
    for i in range(n_centers):
        quad_image_2=image_2[quadrants[i,0]:quadrants[i,1],quadrants[i,2]:quadrants[i,3]]
         
         # plt.figure(i)
         # plt.imshow(quad_image_2)
        centers=np.unravel_index(np.argmax(quad_image_2), quad_image_2.shape)

        centers_coord[i,:]=centers
        centers_coord[i,0]=centers_coord[i,0]+quadrants[i,0]
        centers_coord[i,1]=centers_coord[i,1]+quadrants[i,2]
         #maybe this is redundant now that I implemented  quadrants
        image_2[int(centers_coord[i,0]-radius):int(centers_coord[i,0]+radius),int(centers_coord[i,1]-radius):int(centers_coord[i,1]+radius)]=0
         # 
        if(showFlag==True):
            plt.figure('ROI'+ str(i))
            plt.imshow(image_2)
         
    return centers_coord  
#-------------------------------------------------------------------------------
def find_ROI_tomeasure (radius_ROI,centers_coord,image=None):
    n_ROIs=centers_coord.shape[0]
    ROI=np.zeros((n_ROIs,4), dtype=np.uint16)
    for n in range(n_ROIs):

        ROI[n,:]=([int(centers_coord[n,0]-radius_ROI),int(centers_coord[n,0]+radius_ROI),int(centers_coord[n,1]-radius_ROI),int(centers_coord[n,1]+radius_ROI)]) 
        if (image.any()!=None): 
            plt.figure('ROI'+ str(n))
            plt.imshow(image[ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]]) 
            plt.show()
        
            # offset=50
            # noise=np.sum(image[(ROI[0,0]+offset):(ROI[0,1]+offset), (ROI[0,2]+offset):(ROI[0,3]+offset)])
    return ROI

#-------------------------------------------------------------------------------
def find_ROI_toacquire (radius_roi,centers_original,image):
    roi = np.zeros((centers_original.shape[0],radius_roi*2,radius_roi*2), dtype = np.uint16)
    n_rois=centers_original.shape[0]
   
    for n in range(n_rois):
        print(n)
        roi[n,:,:]=image[int(centers_original[n,0]-radius_roi):int(centers_original[n,0]+radius_roi),int(centers_original[n,1]-radius_roi):int(centers_original[n,1]+radius_roi)]
        plt.figure('roi to acquire'+ str(n))
        plt.imshow(roi[n,:,:]) 
        plt.show()
        # if(saveFlag==True): 
        #     im=Image.fromarray(roi)
            # im.save(save_path+n.zfill(4)+'roi'+'.tif')    
    return roi 
#-------------------------------------------------------------------------------
def save_ROI_original (roi,save_path):
    # os.chdir(save_path)
    subfolder_path=[]
    for i in range(roi.shape[0]):
        im=Image.fromarray(roi[i,:,:])
        subfolder_path.append(save_path+str(i).zfill(5)+'\\')
        os.mkdir(subfolder_path[i])
        im.save(subfolder_path[i]+'original.tif')    
    return subfolder_path
    
#-------------------------------------------------------------------------------
def save_ROI_corrected (roi,subfolder_path):
    # os.chdir(save_path)
    for i in range(roi.shape[0]):
        im=Image.fromarray(roi[i,:,:])
        im.save(subfolder_path[i]+'corrected.tif')    
    return None
#-------------------------------------------------------------------------------

def search_defocus_currents(camera,ETL,start_value,steps_scan,delta_defocus,\
                              ROI,average,save_path,name_holo,roi_height, roi_width,\
                              roi_radius,save_data=False):
    
    # minimi=np.array([])
    second_moments=np.zeros((steps_scan),dtype=np.float32)
    current_nuova = np.zeros((steps_scan,1),dtype=np.float) # all currents corresponding to the optimized images 
    image=np.zeros((roi_height, roi_width,steps_scan),dtype=np.float32)
    current_start=np.copy(start_value)
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range (steps_scan):
        #if loop to give time to the mirror to go from relax to 60 Volt 
        if i==0:
            ETL.current(start_value)
            ##check
            # current_now=o.current(start_value)
            # print(i, current_now)
            sleep(.1)
        else:
            current_start-=delta_defocus
            ETL.current(current_start)
            ##check
            # current_now=o.current(current_start)
            # print(i,current_now)
            sleep(.1)
        
        temp_image = np.zeros((roi_height, roi_width), dtype = np.float32)
        for k in range (average):
            camera.mmc.clearCircularBuffer()
            while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                pass
            camera.mmc.clearCircularBuffer()
            while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                pass
            
            temp_image += camera.mmc.popNextImage()/float(average)
        # temp_image=np.asarray(temp_image)
        image[:,:,i]=temp_image   
        temp_image = temp_image[ROI[0]:ROI[1],ROI[2]:ROI[3]]

        second_moments[i] = mtm.secondmoment(temp_image)
        # minimi=np.append(minimi,min_secondmoment_new)
        current_nuova[i,:]=current_start
        # print str('for'), i, mirror.getvoltages(), second_moments[i]
    camera.mmc.stopSequenceAcquisition()
    
    minimo=np.min(second_moments)
    index_minimo=np.argmin(second_moments)
    ideal_current= current_nuova[index_minimo]
    z_piano = image[:,:,index_minimo]
    I_max=np.amax(z_piano)
    I_min=np.amin(z_piano)
    center=np.unravel_index(np.argmax(z_piano),z_piano.shape)
    ROI_new=np.asarray([center[0]-roi_radius,center[0]+roi_radius,center[1]-roi_radius,center[1]+roi_radius])
    if save_data==True:
        #write o file min second moment
        f= open(save_path+'readme_'+name_holo+'.txt',"a")
        f.write("min second moment %f\r\n" % minimo)
        f.close
        #save image plane in focus
        im=Image.fromarray(z_piano)
        im.save(save_path+'image_'+name_holo+'.tif')
        #save in npy second_moments
        np.save(save_path+'second_moments_'+name_holo,second_moments)
        #plot second moments 
        fig=plt.figure('Second moments'+name_holo)
        plt.plot(second_moments, 'bs-')
        plt.xlabel('Iterations')
        plt.ylabel('Second moment (pixels)')
        plt.title(name_holo)
        # plt.show()
        fig.savefig(save_path+'metric_'+name_holo+'.png', bbox_inches='tight')
        
    return  ideal_current , ROI_new, I_max, I_min
                
#-------------------------------------------------------------------------------
def isocorrection_ROI(SLM,camera,zernikes,weights,syst_zernikes,roi_width,roi_height,ROI,start,save_path,num_zernikes, average_imaq,folder_path='None',name_holo='None',save_flag=True, vol_flag=False ):
    
    
    """ Function to perform isoplatatic correction on one ROI.
    This function applies Zernike polynomials in this order : astigmatism,coma,
    trefoil,spherical. It is useful to apply when we want to correct from a start
    zernike till the last element of the zernikes vector initialized in the main 
    
    Defocus,oblique quadrifoil and oblique secondary astigmatism are skipped
         
    """
    intermediate_roi=np.zeros((num_zernikes,len(weights), ROI[3]-ROI[2], ROI[1]-ROI[0]), dtype= np.float64)
    intermediate_image=np.zeros((num_zernikes,len(weights), roi_height, roi_width), dtype= np.float64)
    print(intermediate_image.shape)
    #initialization data to save 
    saved_ints =  np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    error_ints = np.zeros((num_zernikes, len(weights)), dtype=np.float64)
    dynamic_center= np.zeros((num_zernikes, len(weights),2),dtype= np.uint16)
    final_center=np.zeros((num_zernikes,2),dtype= np.uint16)
    
    # matrix_new = np.copy(pattern)
    # intensity = np.zeros((len(zernikes)-start, len(weights)), dtype = np.uint64)
    # delta_fit=7
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range(start,len(zernikes)):
        print (i)
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        weights=weights-syst_zernikes[i]
        print(weights)
        for j in range(len(weights)):
            image = np.zeros((roi_height, roi_width), dtype = np.float64)
            zernikes[i]=weights[j]
            SLM.setIsoAberration(zernikes)
            sleep(.4)
         
            for k in range (average_imaq):
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            if(i==start and j==0):
                print(type(image[1,1]))
            intermediate_image[i-start,j,:,:]=image
            intermediate_roi[i-start,j,:,:]=image[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            
            center=np.unravel_index(np.argmax(image),image.shape)
            dynamic_center[i-start,j]=center[:]
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im_roi=Image.fromarray(intermediate_roi[i-start,j,:,:])
                if(vol_flag==True):
                   im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
                else: 
                    im.save(save_path+'im_zerny_'+str(i)+'w'+str(j)+'.tif') 
                    im_roi.save(save_path+'roi_zerny_'+str(i)+'w'+str(j)+'.tif')
         
            saved_ints[i-start,j] = np.sum(image[ROI[0]:ROI[1], ROI[2]:ROI[3]])
            
            error_ints[i-start,j]=np.sum(np.sqrt(image[ROI[0]:ROI[1], ROI[2]:ROI[3]]))
    
        
     
    
        # saved_ints[i-start,:] = intensity[i-start,:]   # maybe redundant 

        j_max=np.argmax(saved_ints[i-start,:])
        if (save_flag==True):
            im=Image.fromarray(intermediate_image[i-start,j_max,:,:])
            if(vol_flag==True):
                im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
            else: 
                im.save(save_path+'corrected_z'+str(i)+'.tif')
        
        final_center[i-start,:]=dynamic_center[i-start,j_max]
        zernikes[i]=weights[j_max]
        SLM.setIsoAberration(zernikes)
        print(zernikes)

    # save data
    
      
    camera.mmc.stopSequenceAcquisition()
    if vol_flag==True:
        im.save(folder_path+'corrected.tif')  
        pickle.dump(zernikes, open(save_path+name_holo+"_best_weights.p", "wb"))
        pickle.dump(saved_ints, open(save_path+name_holo+"_ints.p", "wb"))
        pickle.dump(error_ints, open(save_path+name_holo+"_err_ints.p", "wb"))
        
    else:
        im.save(save_path+'corrected.tif')  
        pickle.dump(zernikes, open(save_path+"best_weights.p", "wb"))
        pickle.dump(saved_ints, open(save_path+"ints.p", "wb"))
        pickle.dump(error_ints, open(save_path+"err_ints.p", "wb"))
        pickle.dump(dynamic_center, open(save_path+"dynamic_centers.p", "wb"))
        pickle.dump(final_center, open(save_path+"final_centers.p", "wb"))
        
    im=Image.fromarray(intermediate_image[num_zernikes-1,j_max,:,:])
    
    return zernikes, saved_ints, error_ints ,  dynamic_center, final_center 
#------------------------------------------------------------------------------
def isocorrection_dynamic_ROI(SLM,camera,zernikes,weights,roi_width,roi_height,ROI,radius,start,save_path,num_zernikes, average_imaq,folder_path='None',name_holo='None',save_flag=True, vol_flag=False ):
    
    
    """ Function to perform isoplatatic correction on one ROI.
    This function applies Zernike polynomials in this order : astigmatism,coma,
    trefoil,spherical. It is useful to apply when we want to correct from a start
    zernike till the last element of the zernikes vector initialized in the main 
    
    Defocus,oblique quadrifoil and oblique secondary astigmatism are skipped
         
    """
    intermediate_roi=np.zeros((num_zernikes,len(weights), ROI[3]-ROI[2], ROI[1]-ROI[0]), dtype= np.float64)
    intermediate_image=np.zeros((num_zernikes,len(weights), roi_height, roi_width), dtype= np.float64)
    print(intermediate_image.shape)
    #initialization data to save 
    saved_ints =  np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    error_ints = np.zeros((num_zernikes, len(weights)), dtype=np.float64)
    dynamic_center= np.zeros((num_zernikes, len(weights),2),dtype= np.uint16)
    final_center=np.zeros((num_zernikes,2),dtype= np.uint16)
 
    # matrix_new = np.copy(pattern)
    # intensity = np.zeros((len(zernikes)-start, len(weights)), dtype = np.uint64)
    # delta_fit=7
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range(start,len(zernikes)):
        print (i)
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        for j in range(len(weights)):
            image = np.zeros((roi_height, roi_width), dtype = np.float64)
            zernikes[i]=weights[j]
            SLM.setIsoAberration(zernikes)
            sleep(.4)
         
            for k in range (average_imaq):
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            if(i==start and j==0):
                print(type(image[1,1]))
            intermediate_image[i-start,j,:,:]=image
            
            
            center=np.unravel_index(np.argmax(image),image.shape)
    
            ROI=np.asarray([center[0]-radius,center[0]+radius,center[1]-\
                                                    radius,center[1]+radius])
            dynamic_center[i-start,j]=center[:]
            # print(i,j,center)
            intermediate_roi[i-start,j,:,:]=image[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im_roi=Image.fromarray(intermediate_roi[i-start,j,:,:])
                if(vol_flag==True):
                   im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
                else: 
                    im.save(save_path+'im_zerny_'+str(i)+'w'+str(j)+'.tif') 
                    im_roi.save(save_path+'roi_zerny_'+str(i)+'w'+str(j)+'.tif')
                    
            saved_ints[i-start,j] = np.sum(image[ROI[0]:ROI[1], ROI[2]:ROI[3]])
            
            error_ints[i-start,j]=np.sum(np.sqrt(image[ROI[0]:ROI[1], ROI[2]:ROI[3]]))
    
        
     
    
        # saved_ints[i-start,:] = intensity[i-start,:]   # maybe redundant 

        j_max=np.argmax(saved_ints[i-start,:])
        if (save_flag==True):
            im=Image.fromarray(intermediate_image[i-start,j_max,:,:])
            if(vol_flag==True):
                im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
            else: 
                im.save(save_path+'corrected_z'+str(i)+'.tif')
        
        final_center[i-start,:]=dynamic_center[i-start,j_max]
        zernikes[i]=weights[j_max]
        SLM.setIsoAberration(zernikes)
        print(i,zernikes)

    # save data
    
      
    camera.mmc.stopSequenceAcquisition()
    if vol_flag==True:
        im.save(folder_path+'corrected.tif')  
        pickle.dump(zernikes, open(save_path+name_holo+"_best_weights.p", "wb"))
        pickle.dump(saved_ints, open(save_path+name_holo+"_ints.p", "wb"))
        pickle.dump(error_ints, open(save_path+name_holo+"_err_ints.p", "wb"))
        
    else:
        im.save(save_path+'corrected.tif')  
        pickle.dump(zernikes, open(save_path+"best_weights.p", "wb"))
        pickle.dump(saved_ints, open(save_path+"ints.p", "wb"))
        pickle.dump(error_ints, open(save_path+"err_ints.p", "wb"))
        pickle.dump(dynamic_center, open(save_path+"dynamic_centers.p", "wb"))
        pickle.dump(final_center, open(save_path+"final_centers.p", "wb"))
    
    im=Image.fromarray(intermediate_image[num_zernikes-1,j_max,:,:])
    
    return zernikes, saved_ints, error_ints, dynamic_center, final_center 


#-------------------------------------------------------------------------------
def isocorrection_ROI_bis(SLM,camera,zernikes,weights,syst_zernikes,roi_width,roi_height,ROI,start,save_path,num_zernikes, average_imaq,stop,folder_path='None',name_holo='None',save_flag=True, vol_flag=False ):
    
    
    """  Function to perform isoplatatic correction on one ROI.
    This function applies Zernike polynomials in this order : astigmatism,
    trefoil,spherical. It is useful to apply when we want to correct from a start
    zernike till a stop zernike  
    
    Defocus,oblique quadrifoil and oblique secondary astigmatism are skipped
         
    """
    intermediate_roi=np.zeros((num_zernikes,len(weights), ROI[3]-ROI[2], ROI[1]-ROI[0]), dtype= np.float64)
    intermediate_image=np.zeros((num_zernikes,len(weights), roi_height, roi_width), dtype= np.float64)
    print(intermediate_image.shape)
    #initialization data to save 
    saved_ints =  np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    error_ints = np.zeros((num_zernikes, len(weights)), dtype=np.float64)
    dynamic_center= np.zeros((num_zernikes, len(weights),2),dtype= np.uint16)
    final_center=np.zeros((num_zernikes,2),dtype= np.uint16)
    
    # matrix_new = np.copy(pattern)
    # intensity = np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    # delta_fit=7
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range(start,stop):
        print (i)
        
        if i==4:
            continue
        #skipping coma 
        if i==7:
            continue
        if i==8:
            continue
        if i==10:
            continue
        if i==11:
            continue
        weights=weights-syst_zernikes[i]
        print(weights)
        for j in range(len(weights)):
            image = np.zeros((roi_height, roi_width), dtype = np.float64)
            zernikes[i]=weights[j]
            SLM.setIsoAberration(zernikes)
            sleep(.4)
         
            for k in range (average_imaq):
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            if(i==start and j==0):
                print(type(image[1,1]))
            intermediate_image[i-start,j,:,:]=image
            intermediate_roi[i-start,j,:,:]=image[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            
            center=np.unravel_index(np.argmax(image),image.shape)
            dynamic_center[i-start,j]=center[:]
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im_roi=Image.fromarray(intermediate_roi[i-start,j,:,:])
                if(vol_flag==True):
                   im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
                else: 
                    im.save(save_path+'im_zerny_'+str(i)+'w'+str(j)+'.tif') 
                    im_roi.save(save_path+'roi_zerny_'+str(i)+'w'+str(j)+'.tif')
                     
         
            saved_ints[i-start,j] = np.sum(image[ROI[0]:ROI[1], ROI[2]:ROI[3]])
            
            error_ints[i-start,j]=np.sum(np.sqrt(image[ROI[0]:ROI[1], ROI[2]:ROI[3]]))
    
        
     
    
        # saved_ints[i-start,:] =  saved_ints[i-start,:]   # maybe redundant 
      
        j_max=np.argmax(saved_ints[i-start,:])
        if (save_flag==True):
            im=Image.fromarray(intermediate_image[i-start,j_max,:,:])
            if(vol_flag==True):
                im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
            else: 
                im.save(save_path+'corrected_z'+str(i)+'.tif')
        
        final_center[i-start,:]=dynamic_center[i-start,j_max]
        zernikes[i]=weights[j_max]
        SLM.setIsoAberration(zernikes)
        print(i,zernikes)

    # save data
    
      
    camera.mmc.stopSequenceAcquisition()
    if vol_flag==True:
        im.save(folder_path+'corrected_bis.tif')  
        pickle.dump(zernikes, open(save_path+name_holo+"_best_weights_bis.p", "wb"))
        pickle.dump(saved_ints, open(save_path+name_holo+"_ints_bis.p", "wb"))
        pickle.dump(error_ints, open(save_path+name_holo+"_err_ints_bis.p", "wb"))
        
    else:
        im.save(save_path+'corrected_bis.tif')  
        pickle.dump(zernikes, open(save_path+"best_weights_bis.p", "wb"))
        pickle.dump(saved_ints, open(save_path+"ints_bis.p", "wb"))
        pickle.dump(error_ints, open(save_path+"err_ints_bis.p", "wb"))
        pickle.dump(dynamic_center, open(save_path+"dynamic_centers_bis.p", "wb"))
        pickle.dump(final_center, open(save_path+"final_centers_bis.p", "wb"))
 
    im=Image.fromarray(intermediate_image[num_zernikes-1,j_max,:,:])
    return zernikes, saved_ints, error_ints , dynamic_center, final_center
#-------------------------------------------------------------------------------
def isocorrection_dynamic_ROI_bis(SLM,camera,zernikes,weights,roi_width,roi_height,ROI,radius,start,save_path,num_zernikes, average_imaq,stop,folder_path='None',name_holo='None',save_flag=True, vol_flag=False ):   
    
    """  Function to perform isoplatatic correction on one ROI.
    This function applies Zernike polynomials in this order : astigmatism,coma,
    trefoil,spherical. It is useful to apply when we want to correct from a start
    zernike till a stop zernike  
    
    Defocus,oblique quadrifoil and oblique secondary astigmatism are skipped
         
    """
    intermediate_roi=np.zeros((num_zernikes,len(weights), ROI[3]-ROI[2], ROI[1]-ROI[0]), dtype= np.float64)
    intermediate_image=np.zeros((num_zernikes,len(weights), roi_height, roi_width), dtype= np.float64)
    print(intermediate_image.shape)
    #initialization data to save 
    saved_ints =  np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    error_ints = np.zeros((num_zernikes, len(weights)), dtype=np.float64)
    dynamic_center= np.zeros((num_zernikes, len(weights),2),dtype= np.uint16)
    final_center=np.zeros((num_zernikes,2),dtype= np.uint16)
    
    # matrix_new = np.copy(pattern)
    # intensity = np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    # delta_fit=7
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range(start,stop):
        print (i)
        
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        for j in range(len(weights)):
            image = np.zeros((roi_height, roi_width), dtype = np.float64)
            zernikes[i]=weights[j]
            SLM.setIsoAberration(zernikes)
            sleep(.4)
         
            for k in range (average_imaq):
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            if(i==start and j==0):
                print(type(image[1,1]))
            intermediate_image[i-start,j,:,:]=image
            
            
            center=np.unravel_index(np.argmax(image),image.shape)
    
            ROI=np.asarray([center[0]-radius,center[0]+radius,center[1]-\
                                                    radius,center[1]+radius])
            dynamic_center[i-start,j]=center[:]
            intermediate_roi[i-start,j,:,:]=image[ROI[0]:ROI[1], ROI[2]:ROI[3]]
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im_roi=Image.fromarray(intermediate_roi[i-start,j,:,:])
                if(vol_flag==True):
                   im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
                else: 
                    im.save(save_path+'im_zerny_'+str(i)+'w'+str(j)+'.tif') 
                    im_roi.save(save_path+'roi_zerny_'+str(i)+'w'+str(j)+'.tif')
         
            saved_ints[i-start,j] = np.sum(image[ROI[0]:ROI[1], ROI[2]:ROI[3]])
            
            error_ints[i-start,j]=np.sum(np.sqrt(image[ROI[0]:ROI[1], ROI[2]:ROI[3]]))
    
        
     
    
        # saved_ints[i-start,:] =  saved_ints[i-start,:]   # maybe redundant 

        j_max=np.argmax(saved_ints[i-start,:])
        
        if (save_flag==True):
            im=Image.fromarray(intermediate_image[i-start,j_max,:,:])
            if(vol_flag==True):
                im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
            else: 
                im.save(save_path+'corrected_z'+str(i)+'.tif')
        
        final_center[i-start,:]=dynamic_center[i-start,j_max]
        zernikes[i]=weights[j_max]
        SLM.setIsoAberration(zernikes)
        print(zernikes)

    # save data
    
      
    camera.mmc.stopSequenceAcquisition()
    if vol_flag==True:
        im.save(folder_path+'corrected_bis.tif')  
        pickle.dump(zernikes, open(save_path+name_holo+"_best_weights_bis.p", "wb"))
        pickle.dump(saved_ints, open(save_path+name_holo+"_ints_bis.p", "wb"))
        pickle.dump(error_ints, open(save_path+name_holo+"_err_ints_bis.p", "wb"))
        
    else:
        im.save(save_path+'corrected_bis.tif')  
        pickle.dump(zernikes, open(save_path+"best_weights_bis.p", "wb"))
        pickle.dump(saved_ints, open(save_path+"ints_bis.p", "wb"))
        pickle.dump(error_ints, open(save_path+"err_ints_bis.p", "wb"))
        pickle.dump(dynamic_center, open(save_path+"dynamical_centers_bis.p", "wb"))
        pickle.dump(final_center, open(save_path+"final_centers_bis.p", "wb"))
        
    im=Image.fromarray(intermediate_image[num_zernikes-1,j_max,:,:])
    return zernikes, saved_ints, error_ints,dynamic_center, final_center   
#-------------------------------------------------------------------------------
def isocorrection_ROI_tris(SLM,camera,zernikes,weights,roi_width,roi_height,ROI,start,save_path,num_zernikes, average_imaq,stop,folder_path='None',name_holo='None',save_flag=True, vol_flag=False ):
    
    
    """Function to perform isoplatatic correction on one ROI.
    This function applies Zernike polynomials in this order : astigmatism,coma,
    trefoil,spherical. It is useful to apply when we want to correct from a start
    zernike till a stop zernike and when the trefoil will be applied one after 
    the others since comas are skipped
    
    Defocus,comas, oblique quadrifoil and oblique secondary astigmatism are skipped
    """
    
    intermediate_image=np.zeros((num_zernikes,len(weights), roi_height, roi_width), dtype= np.float64)
    print(intermediate_image.shape)
    #initialization data to save 
    saved_ints =  np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    error_ints = np.zeros((num_zernikes, len(weights)), dtype=np.float64)
    
    # matrix_new = np.copy(pattern)
    # intensity = np.zeros((num_zernikes, len(weights)), dtype = np.uint64)
    # delta_fit=7
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    for i in range(start,stop):
        # print (i)
        # print(i-start)
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        if i==7:
            continue
        if i==8:
            continue
        m=0
        print (i, m)
        for j in range(len(weights)):
            image = np.zeros((roi_height, roi_width), dtype = np.float64)
            zernikes[i]=weights[j]
            SLM.setIsoAberration(zernikes)
            sleep(.4)
         
            for k in range (average_imaq):
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            intermediate_image[m,j,:,:]=image
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[m,j,:,:])
                if(vol_flag==True):
                   im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
                else: 
                    im.save(save_path+'zerny_'+str(i)+'w'+str(j)+'.tif') 
         
            saved_ints[m,j] = np.sum(image[ROI[0]:ROI[1], ROI[2]:ROI[3]])
            
            error_ints[m,j]=np.sum(np.sqrt(image[ROI[0]:ROI[1], ROI[2]:ROI[3]]))
    
        
     
    
        # saved_ints[i-start,:] = intensity[i-start,:]   # maybe redundant 

        j_max=np.argmax(saved_ints[m,:])
        if (save_flag==True):
            im=Image.fromarray(intermediate_image[m,j_max,:,:])
            if(vol_flag==True):
                im.save(save_path+name_holo+'corrected_zerny_'+str(i)+'.tif')
            else: 
                im.save(save_path+'corrected_z'+str(i)+'.tif')
        
        
        zernikes[i]=weights[j_max]
        SLM.setIsoAberration(zernikes)
        print(zernikes)
        m=m+1
        
    camera.mmc.stopSequenceAcquisition()
    
    # save data
    if vol_flag==True:
        im.save(folder_path+'corrected_tris.tif')  
        pickle.dump(zernikes, open(save_path+name_holo+"_best_weights_tris.p", "wb"))
        pickle.dump(saved_ints, open(save_path+name_holo+"_ints_tris.p", "wb"))
        pickle.dump(error_ints, open(save_path+name_holo+"_err_ints_tris.p", "wb"))
        
    else:
        im.save(save_path+'corrected_tris.tif')  
        pickle.dump(zernikes, open(save_path+"best_weights_tris.p", "wb"))
        pickle.dump(saved_ints, open(save_path+"ints_tris.p", "wb"))
        pickle.dump(error_ints, open(save_path+"err_ints_tris.p", "wb"))
    
    im=Image.fromarray(intermediate_image[num_zernikes-1,j_max,:,:])
    
    return zernikes, saved_ints, error_ints       
#-------------------------------------------------------------------------------    
def app_isocorrection_2D_syst(ETL, SLM, camera, semi_path,save_path,file_name, current_value,index_pattern=0):

    best_zernikes=pickle.load(open(save_path+file_name, "rb"))
    print(best_zernikes)
    ETL.current(current_value)
    holo=np.load(semi_path)
    print ("setting phase")
    SLM.setPhase(holo)
    sleep(.4) 
    
    #acquisition original image 
    camera.snap(False)
    im_original=Image.fromarray(camera.snapped)
    im_original.save(save_path+'total_original.tif')
    
    #acquisition corrected image 
    SLM.setIsoAberration(best_zernikes)
    sleep(.4)
    camera.snap(False)
    im_corrected=Image.fromarray(camera.snapped)
    im_corrected.save(save_path+'total_corrected.tif')

    return None
#-------------------------------------------------------------------------------
def app_isocorrection_2D_sample(ETL, SLM, camera, semi_path,save_path,file_name_syst,file_name_sample, current_value,index_pattern=0):

    best_zernikes_syst=pickle.load(open(file_name_syst, "rb"))
    print(best_zernikes_syst)
    best_zernikes_sample=pickle.load(open(file_name_sample, "rb"))
    print(best_zernikes_sample)
    ETL.current(current_value)
    holo=np.load(semi_path)
    print ("setting phase")
    SLM.setPhase(holo)
    sleep(.4) 
    
    #acquisition original image 
    SLM.setIsoAberration(best_zernikes_syst)
    sleep(.4)
    camera.snap(False)
    im_original=Image.fromarray(camera.snapped)
    im_original.save(save_path+'total_original.tif')
    
    #acquisition corrected image 
    SLM.setIsoAberration(best_zernikes_sample)
    sleep(.4)
    camera.snap(False)
    im_corrected=Image.fromarray(camera.snapped)
    im_corrected.save(save_path+'total_corrected.tif')

    return None
#------------------------------------------------------------------------------
def aniso_correction_dynamicROI (SLM,camera,zernikes,num_zernikes,aniso_zernikes,weights,average_imaq,points_number,\
                ROI,centers_original,radius_roi,start,rad_ROI,coords,save_path,save_flag=False, vol_flag=False):
    
    #initialization data to save 
    # error_ints = np.zeros((len(zernikes)-start, weights.shape[1],points_number), dtype=np.uint64)
    # intensity = np.zeros((len(zernikes)-start, weights.shape[1],points_number), dtype = np.uint64)
    # intermediate_image=np.zeros((num_zernikes,weights.shape[1],camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype= np.float64)
    # temp_im=np.zeros((points_number,num_zernikes,weights.shape[1],radius_roi*2,radius_roi*2), dtype= np.float64)
    # dynamic_center= np.zeros((num_zernikes, weights.shape[1],points_number,2),dtype= np.uint16)
    error_ints = np.zeros((len(zernikes)-start, weights.shape[0],points_number), dtype=np.uint64)
    intensity = np.zeros((len(zernikes)-start, weights.shape[0],points_number), dtype = np.uint64)
    intermediate_image=np.zeros((num_zernikes,weights.shape[0],camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype= np.float64)
    temp_im=np.zeros((points_number,num_zernikes,weights.shape[0],radius_roi*2,radius_roi*2), dtype= np.float64)
    dynamic_center= np.zeros((num_zernikes, weights.shape[0],points_number,2),dtype= np.uint16)
    final_center= np.zeros((num_zernikes,points_number,2),dtype= np.uint16)
    # roi_image=
                
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    #for loop over poly
    for i in range(start,aniso_zernikes.shape[1]):
        # print ('zernike',i,flush=True)
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        
        # for loop over the weights
        # for j in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            image = np.zeros((camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype = np.float64)
            # print ('w',j,flush=True)
            # zernikes[i]=weights[i,j] 
            zernikes[i]=weights[j]# apllying same aberration to 4 different points 
            # print(zernikes)
            SLM.setIsoAberration(zernikes)
            sleep(.4)
            
            
            #for loop to acquire averged images
            for k in range (average_imaq):
                    # print(k,flush=True)
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            intermediate_image[i-start,j,:,:]=image
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im.save(save_path+'zerny_'+str(i)+'w'+str(j)+'.tif')
            
            for n in range(points_number):
                temp_im[n,i-start,j,:,:]=image[int(centers_original[n,0]-radius_roi):int(centers_original[n,0]+radius_roi),int(centers_original[n,1]-radius_roi):int(centers_original[n,1]+radius_roi)]
                center=np.unravel_index(np.argmax(temp_im[n,i-start,j,:,:]),temp_im[n,i-start,j,:,:].shape)
                roi_image=Image.fromarray(temp_im[n,i-start,j,:,:])
                roi_image.save(save_path+'im_interm'+str(n).zfill(3)+'_'+str(i)+'_'+str(j).zfill(5)+'.tif')
                
                dynamic_center[i-start,j,n,:]=center
                
                #calculation dynamic ROI to measure 
                ROI[n,:]=([int(center[0]-rad_ROI),int(center[0]+rad_ROI),int(center[1]-rad_ROI),int(center[1]+rad_ROI)])
            
            
            # for loop over ROIs to measure Intensity
            for n in range(len(ROI)):
                
                # print ('roi',n,flush=True)
                
                # 
                # intensity[i-start,j,n] = np.sum(image[ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                # 
                # error_ints[i-start,j,n]=np.sum(np.sqrt(image[ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]]))
                ROI_im=Image.fromarray(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                ROI_im.save(save_path+'ROI_interm'+str(n).zfill(3)+'_'+str(i)+'_'+str(j).zfill(5)+'.tif')
                intensity[i-start,j,n] = np.sum(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                # 
                error_ints[i-start,j,n]=np.sum(np.sqrt(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]]))
                
            
        # calculation of best weights for each zernike for each roi 
        #new holo with aniso correction for the i zernike
        for h in range(len(ROI)):
            j_max=np.argmax(intensity[i-start,:,h])
            # aniso_zernikes[h,i]=weights[i,j_max]
            aniso_zernikes[h,i]=weights[j_max]
            final_center[i-start,h,:]=dynamic_center[i-start,j_max,h]
            print(i,h,j_max,aniso_zernikes[h,i])
            
            # save roi corrected for each zernike
            im=Image.fromarray(temp_im[n,i-start,j_max,:,:])
            im.save(save_path+'ROI_'+str(h)+'zerny_'+str(i)+'.tif')
        
        # apply aniosocorrections 
        print(aniso_zernikes)
        holo=hologram(coords,aniso_zernikes,SLM,0,0)
        holo.compute("RS")
        # holo.compute("RS")
        holo.wait()

        # print('setting Phase '+str(i))
        SLM.setPhase(holo.phase)
        sleep(.4)
        np.save(save_path+str(i)+'_rs_holo',holo.phase)
        zernikes=np.zeros((13), dtype=np.float)
        
            
    camera.mmc.stopSequenceAcquisition()
    pickle.dump(aniso_zernikes, open(save_path+"_best_weights.p", "wb"))
    pickle.dump(intensity, open(save_path+"_ints.p", "wb"))
    pickle.dump(error_ints, open(save_path+"_err_ints.p", "wb"))
    pickle.dump(dynamic_center, open(save_path+"dynamical_centers.p", "wb"))
    pickle.dump(final_center, open(save_path+"final_centers.p", "wb"))
    
    return aniso_zernikes,intensity, error_ints  , final_center
# 
# #------------------------------------------------------------------------------
def aniso_correction_dynamicROI_bis (SLM,camera,zernikes,num_zernikes,aniso_zernikes,weights,average_imaq,points_number,\
                ROI,centers_original,radius_roi,start,stop,rad_ROI,coords,save_path,save_flag=False, vol_flag=False):
    
    #initialization data to save 
    # error_ints = np.zeros((len(zernikes)-start, weights.shape[1],points_number), dtype=np.uint64)
    # intensity = np.zeros((len(zernikes)-start, weights.shape[1],points_number), dtype = np.uint64)
    # intermediate_image=np.zeros((num_zernikes,weights.shape[1],camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype= np.float64)
    # temp_im=np.zeros((points_number,num_zernikes,weights.shape[1],radius_roi*2,radius_roi*2), dtype= np.float64)
    # dynamic_center= np.zeros((num_zernikes, weights.shape[1],points_number,2),dtype= np.uint16)
    error_ints = np.zeros((num_zernikes, weights.shape[0],points_number), dtype=np.uint64)
    intensity = np.zeros((num_zernikes, weights.shape[0],points_number), dtype = np.uint64)
    intermediate_image=np.zeros((num_zernikes,weights.shape[0],camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype= np.float64)
    temp_im=np.zeros((points_number,num_zernikes,weights.shape[0],radius_roi*2,radius_roi*2), dtype= np.float64)
    dynamic_center= np.zeros((num_zernikes, weights.shape[0],points_number,2),dtype= np.uint16)
    final_center= np.zeros((num_zernikes,points_number,2),dtype= np.uint16)
    # roi_image=
                
    
    camera.mmc.startContinuousSequenceAcquisition(0)
    #for loop over poly
    for i in range(start,stop):
        # print ('zernike',i,flush=True)
        if i==4:
            continue
        if i==10:
            continue
        if i==11:
            continue
        
        # for loop over the weights
        # for j in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            image = np.zeros((camera.mmc.getImageHeight(),camera.mmc.getImageWidth()), dtype = np.float64)
            # print ('w',j,flush=True)
            # zernikes[i]=weights[i,j] 
            zernikes[i]=weights[j]# apllying same aberration to 4 different points 
            # print(zernikes)
            SLM.setIsoAberration(zernikes)
            sleep(.4)
            
            
            #for loop to acquire averged images
            for k in range (average_imaq):
                    # print(k,flush=True)
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                camera.mmc.clearCircularBuffer()
                while camera.mmc.getBufferTotalCapacity()-camera.mmc.getBufferFreeCapacity()==0:
                    pass
                image += camera.mmc.popNextImage()/float(average_imaq)
            intermediate_image[i-start,j,:,:]=image
            
            if (save_flag==True):
                im=Image.fromarray(intermediate_image[i-start,j,:,:])
                im.save(save_path+'zerny_'+str(i)+'w'+str(j)+'.tif')
            
            for n in range(points_number):
                temp_im[n,i-start,j,:,:]=image[int(centers_original[n,0]-radius_roi):int(centers_original[n,0]+radius_roi),int(centers_original[n,1]-radius_roi):int(centers_original[n,1]+radius_roi)]
                center=np.unravel_index(np.argmax(temp_im[n,i-start,j,:,:]),temp_im[n,i-start,j,:,:].shape)
                roi_image=Image.fromarray(temp_im[n,i-start,j,:,:])
                roi_image.save(save_path+'im_interm'+str(n).zfill(3)+'_'+str(i)+'_'+str(j).zfill(5)+'.tif')
                
                dynamic_center[i-start,j,n,:]=center
                
                #calculation dynamic ROI to measure 
                ROI[n,:]=([int(center[0]-rad_ROI),int(center[0]+rad_ROI),int(center[1]-rad_ROI),int(center[1]+rad_ROI)])
            
            
            # for loop over ROIs to measure Intensity
            for n in range(len(ROI)):
                
                # print ('roi',n,flush=True)
                
                # 
                # intensity[i-start,j,n] = np.sum(image[ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                # 
                # error_ints[i-start,j,n]=np.sum(np.sqrt(image[ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]]))
                ROI_im=Image.fromarray(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                ROI_im.save(save_path+'ROI_interm'+str(n).zfill(3)+'_'+str(i)+'_'+str(j).zfill(5)+'.tif')
                intensity[i-start,j,n] = np.sum(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]])
                # 
                error_ints[i-start,j,n]=np.sum(np.sqrt(temp_im[n,i-start,j,ROI[n,0]:ROI[n,1], ROI[n,2]:ROI[n,3]]))
                
            
        # calculation of best weights for each zernike for each roi 
        #new holo with aniso correction for the i zernike
        for h in range(len(ROI)):
            j_max=np.argmax(intensity[i-start,:,h])
            # aniso_zernikes[h,i]=weights[i,j_max]
            aniso_zernikes[h,i]=weights[j_max]
            final_center[i-start,h,:]=dynamic_center[i-start,j_max,h]
            print(i,h,j_max,aniso_zernikes[h,i])
            
            # save roi corrected for each zernike
            im=Image.fromarray(temp_im[n,i-start,j_max,:,:])
            im.save(save_path+'ROI_'+str(h)+'zerny_'+str(i)+'.tif')
        
        # apply aniosocorrections 
        print(aniso_zernikes)
        holo=hologram(coords,aniso_zernikes,SLM,0,0)
        holo.compute("RS")
        # holo.compute("RS")
        holo.wait()

        # print('setting Phase bis'+str(i))
        SLM.setPhase(holo.phase)
        sleep(.4)
        np.save(save_path+str(i)+'_rs_holo_bis',holo.phase)
        zernikes=np.zeros((13), dtype=np.float)
        
            
    camera.mmc.stopSequenceAcquisition()
    pickle.dump(aniso_zernikes, open(save_path+"best_weights_bis.p", "wb"))
    pickle.dump(intensity, open(save_path+"ints_bis.p", "wb"))
    pickle.dump(error_ints, open(save_path+"err_ints_bis.p", "wb"))
    pickle.dump(dynamic_center, open(save_path+"dynamical_centers_bis.p", "wb"))
    pickle.dump(final_center, open(save_path+"final_centers_bis.p", "wb"))
    return aniso_zernikes,intensity, error_ints  , final_center  

#-------------------------------------------------------------------------------

def app_anisocorrection_2D(ETL, SLM, camera, semi_path,save_path,file_name, current_value,index_pattern=0):

    best_zernikes=pickle.load(open(save_path+file_name, "rb"))
    print(best_zernikes)
    ETL.current(current_value)
    holo=np.load(semi_path)
    print ("setting phase")
    SLM.setPhase(holo)
    sleep(.4) 
    
    #acquisition original image 
    camera.snap(False)
    im_original=Image.fromarray(camera.snapped)
    im_original.save(save_path+'total_original.tif')
    
    #acquisition corrected image 
    SLM.setIsoAberration(best_zernikes)
    sleep(.4)
    camera.snap(False)
    im_corrected=Image.fromarray(camera.snapped)
    im_corrected.save(save_path+'total_corrected.tif')

    return None

 
    