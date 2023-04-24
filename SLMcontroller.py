import sys
from multiprocessing import RawArray,Process,cpu_count
import psutil
import numpy
import time
import pygame
from pygame.locals import *
import screeninfo
import os
from scipy.special import factorial as fact
from scipy.linalg import hadamard
import pickle
import PIL.Image


##--------------------------------------------------------------------------------------
## Lut generation function


#redBits, greenBits and blueBits are either 0 (if not used), 1 (first 8 bit), 2 (bits 9 to 16)
#or 3 (bits 17 to 24)

def generateLinearLUT(zeroVal,twoPiVal,redBits,greenBits,blueBits):
    values=numpy.linspace(zeroVal,twoPiVal,2**16).astype("uint32")
    LUTvalues=numpy.zeros(2**16,dtype="uint32")
    if blueBits==0:
        pass
    if blueBits==1:
        LUTvalues+=(values%(256))    
    if blueBits==2:
        LUTvalues+=(values%(65536))/(256)
    if blueBits==3:
        LUTvalues+=(values%(16777216))/(65536)
    if greenBits==0:
        pass
    if greenBits==1:
        LUTvalues+=(values%(256))*256  
    if greenBits==2:
        LUTvalues+=((values%(65536))/(256))*256
    if greenBits==3:
        LUTvalues+=((values%(16777216))/(65536))*256
    if redBits==0:
        pass
    if redBits==1:
        LUTvalues+=(values%(256))*65536 
    if redBits==2:
        LUTvalues+=((values%(65536))/(256))*65536
    if redBits==3:
        LUTvalues+=((values%(16777216))/(65536))*65536

    return LUTvalues


##--------------------------------------------------------------------------------------
##Aberrations Generation Functions (normalized to 1 PV)

def nollIndex(n,m):
    return numpy.sum(numpy.arange(n+1))+numpy.arange(-n,n+1,2).tolist().index(m)
        
def zernikeRadialComponent(n,m,r):
    k=numpy.arange((n-m)/2+1)
    k_facts=((-1)**k*fact(n-k))/(fact(k)*fact((n+m)/2-k)*fact((n-m)/2-k))
    k_facts=numpy.meshgrid(k_facts,numpy.zeros(r.shape))[0]
    k=numpy.meshgrid(k,numpy.zeros(r.shape))[0]
    return numpy.sum(k_facts*r[:,None]**(n-2*k),axis=1)       

def lukoszRadialComponent(n,m,r):
    if m==n:
        return zernikeRadialComponent(n,m,r)
    else:
        return zernikeRadialComponent(n,m,r)-zernikeRadialComponent(n-2,m,r)


def generateAberrationDataset(res,base,order,existingN=0):
    xc,yc=numpy.meshgrid(numpy.linspace(-1.0,1.0,res),numpy.linspace(-1.0,1.0,res))
    r=numpy.sqrt(xc**2+yc**2).flatten()
    pupilCoords=numpy.where(r<=1.0)
    if base=="Zernike" or base=="Lukosz": 
        t=numpy.arctan2(yc,xc).flatten()
        ns,ms=numpy.meshgrid(numpy.arange(0,order+1),numpy.arange(-order,order+1))
        ns_notzero=ns[numpy.where(numpy.logical_and(numpy.abs(ms)<=ns,(ns-ms)%2==0))]
        ms_notzero=ms[numpy.where(numpy.logical_and(numpy.abs(ms)<=ns,(ns-ms)%2==0))]
        dataset=numpy.zeros((res**2,ns_notzero.shape[0]-existingN),dtype="float32")
   
        for i in range(ns_notzero.shape[0]):
            ind=nollIndex(ns_notzero[i],ms_notzero[i])
            if ind>existingN:
                if ns_notzero[i]==0:
                    dataset[:,ind-existingN]=1.0
                else:
                    if base=="Zernike":
                        temp=zernikeRadialComponent(ns_notzero[i],numpy.abs(ms_notzero[i]),r)[pupilCoords]
                    elif base=="Lukosz":
                        temp=lukoszRadialComponent(ns_notzero[i],numpy.abs(ms_notzero[i]),r)[pupilCoords]

                    if ms_notzero[i]>0:
                        temp=(temp*numpy.cos(ms_notzero[i]*t[pupilCoords])).astype("float32")
                    elif ms_notzero[i]<0:
                        temp=(temp*numpy.sin(-ms_notzero[i]*t[pupilCoords])).astype("float32")
                    dataset[pupilCoords,ind-existingN]=((temp-numpy.min(temp))/(numpy.max(temp)-numpy.min(temp)))

    if base=="Hadamard":
##        if res%2==0:
##            realres=res
##        else:
##            realres=
        N=2**order
        hm=hadamard(N).astype("int8")
        indexes=numpy.argsort(numpy.bincount(numpy.where(hm[:,:-1]!=hm[:,1:])[0]))
        hm=hm[indexes,:]
        hm=hm.repeat(int(res/N),axis=0)
        dataset=numpy.zeros((N,N,res,res),dtype="int8")
        dataset[0,0,:,:]=numpy.ones((res,res),dtype="int8")
        indexes=numpy.dstack(numpy.meshgrid(numpy.arange(N), numpy.arange(N))).reshape(-1, 2)
        indexes=indexes[numpy.lexsort((numpy.amin(indexes,axis=1),numpy.amax(indexes,axis=1)))]

        for i in range(N-1):
            dataset[0,i+1,:,:],base[i+1,0,:,:]=numpy.meshgrid(hm[:,i+1],hm[:,i+1])

        for i in range(N-1):
            for j in range(N-1):
                dataset[j+1,i+1,:,:]=dataset[0,i+1,:,:]*base[j+1,0,:,:]

        dataset=dataset[indexes[:,0],indexes[:,1],:,:][pupilcoords].astype("float32")      

    return dataset

def getAberrationPhase(dataset,lam,coeffs):
    return numpy.dot(dataset[:,:coeffs.shape[0]],coeffs.astype("float32")).reshape((numpy.sqrt(dataset.shape[0]).astype("uint32"),numpy.sqrt(dataset.shape[0]).astype("uint32")))/(lam/1000.0)*2**16


#---------------------------------------------------------------------------------------
##Hologram Classes

class hologram:
    def __init__(self,points,intensities,aberrations,playerInst,sourceIndex,apertureIndex):
        self.points=points
        self.aberrations=aberrations
        self.intensities=intensities
        self.playerInst=playerInst
        self.sourceIndex=sourceIndex
        self.apertureIndex=apertureIndex
        self.performance=numpy.zeros(2,dtype="float64")
        self.algorithm=None
        self.phase=numpy.zeros(playerInst.resolution,dtype="uint32")
        self.p=None
    
        print(playerInst.maxTilt(self.points[:,2],sourceIndex,apertureIndex))
        #original from Paolo
        # if numpy.any(self.points[:,0]>playerInst.maxTilt(self.points[:,2],sourceIndex,apertureIndex)) or numpy.any(self.points[:,1]>playerInst.maxTilt(self.points[:,2],sourceIndex,apertureIndex)):
        #     raise ValueError("Lateral isplacement too big for SLM resolution")
        # New  version LM 
        if numpy.any(self.points[:,0]>2*playerInst.maxTilt(0.0,sourceIndex,apertureIndex)) or numpy.any(self.points[:,1]>2*playerInst.maxTilt(0.0,sourceIndex,apertureIndex)):
            raise ValueError("Lateral isplacement too big for SLM resolution")
      

    def compute(self,algorithm,iterations=1,compression=1.0,cpuId=None):
        phaseRaw=RawArray("c",self.playerInst.resolution[0]*self.playerInst.resolution[1]*numpy.dtype("uint32").itemsize)
        self.phase=numpy.frombuffer(phaseRaw, dtype="uint32").reshape(self.playerInst.resolution)
        performanceRaw=RawArray("c",2*numpy.dtype("float64").itemsize)
        self.performance=numpy.frombuffer(performanceRaw, dtype="float64")
        lam=self.playerInst.sources[self.sourceIndex][0]*10**(-9)
        res=self.playerInst.apertures[self.apertureIndex][1]
        corner=self.playerInst.apertures[self.apertureIndex][2]
        pixSize=self.playerInst.pixSize*10**(-6)
        focal=self.playerInst.focal*10**(-3)
        pupilInt=numpy.copy(self.playerInst.sources[self.sourceIndex][1][corner[0]:corner[0]+res,corner[1]:corner[1]+res])
        self.p=Process(target=gsLoopPoints,args=(phaseRaw,self.playerInst.resolution,performanceRaw,self.points,self.intensities,self.aberrations,self.playerInst.abBases[self.playerInst.apertures[self.apertureIndex][3]][3],lam,corner,res,pixSize,focal,pupilInt,algorithm,iterations,compression,cpuId))
        self.p.daemon=True
        self.p.start()
        self.algorithm=algorithm
        
        print('aperture',res,'pixSize',pixSize,'source', self.playerInst.sources[self.sourceIndex][1].shape )

    def wait(self):
        if self.p is not None:
            self.p.join()

def gsLoopPoints(phaseRaw,phaseRes,performanceRaw,points,intensities,aberrations,abDataset,lam,corner,res,pixSize,focal,pupilInt,algorithm,iterations,compression,cpuId):
    p=psutil.Process()
    if cpuId is not None and cpuId<cpu_count():
        p.cpu_affinity([cpuId+2])
    else:
        p.cpu_affinity(range(2,cpu_count()))
    phase=numpy.frombuffer(phaseRaw, dtype="uint32").reshape(phaseRes)
    performance=numpy.frombuffer(performanceRaw, dtype="float64")
    coords=points*(10**(-6))
    weights=numpy.ones(coords.shape[0])/float(coords.shape[0])
    xc,yc=numpy.meshgrid(numpy.linspace(-1.0,1.0,res),numpy.linspace(-1.0,1.0,res))
    xc=xc.flatten()
    yc=yc.flatten()
    pupilCoords=numpy.where(numpy.sqrt(xc**2+yc**2)<=1.0)
    xc=xc*res*pixSize/2.0
    yc=yc*res*pixSize/2.0
    pupilInt=pupilInt.flatten()
    pupilInt=pupilInt[pupilCoords]/numpy.sum(pupilInt[pupilCoords])
    
    print('xc',xc,'yc',yc)
    

    pists=numpy.random.random(coords.shape[0])*2*numpy.pi
    coordsList=numpy.asarray(range(pupilCoords[0].shape[0]))
    coordsListSparse=coordsList
    if algorithm=="CSGS" or algorithm=="CSWGS":
        numpy.random.shuffle(coordsList)

    
    slm_p_phase=numpy.zeros((coords.shape[0],pupilCoords[0].shape[0]),dtype="float32")
    for i in range(coords.shape[0]):
        aberrationPhase=0.0
        if aberrations is not None:
            if aberrations[i] is not None:
                if aberrations[i].shape==(res,res):
                    aberrationPhase=aberration[i].flatten()[pupilCoords]
                elif len(aberrations[i].shape)==1:
                    aberrationPhase=(getAberrationPhase(abDataset,lam*10**9,aberrations[i]).astype(float)/(2**16)*2*numpy.pi).flatten()[pupilCoords]
                else:
                    raise ValueError("aberration data shape mismatch")

                
        slm_p_phase[i,:]=2.0*numpy.pi/(lam*(focal))*(coords[i,0]*xc[pupilCoords]+
            coords[i,1]*yc[pupilCoords])+(numpy.pi*coords[i,2])/(lam*(focal)**2
            )*(xc[pupilCoords]**2+yc[pupilCoords]**2)+aberrationPhase



    if algorithm is not "RS":
        for i in range(iterations):
            if algorithm=="CSGS" or algorithm=="CSWGS":
                coordsList=numpy.roll(coordsList,int(coordsList.shape[0]*compression))
                coordsListSparse=coordsList[:int(coordsList.shape[0]*compression)]

            slm_total_field=numpy.sum((intensities*weights)[:,None]/(float(pupilCoords[0].shape[0]))*numpy.exp(1.0j*(slm_p_phase[:,coordsListSparse]+pists[:,None])),axis=0)
            slm_total_phase=numpy.angle(slm_total_field)

            spot_fields=numpy.sum(pupilInt[None,coordsListSparse]*numpy.exp(1j*(slm_total_phase[None,:]-slm_p_phase[:,coordsListSparse])),axis=1)
            pists=numpy.angle(spot_fields)
            
            
            if algorithm=="WGS":
                ints=numpy.abs(spot_fields)**2
                weights=weights*(numpy.mean(numpy.sqrt(ints/intensities))/numpy.sqrt(ints/intensities))
                weights=weights/numpy.sum(weights)   

    
    slm_total_field=numpy.sum((intensities*weights)[:,None]/(float(pupilCoords[0].shape[0]))*numpy.exp(1.0j*(slm_p_phase+pists[:,None])),axis=0)
    slm_total_phase=numpy.angle(slm_total_field)
    


    spot_fields=numpy.sum(pupilInt[None,:]*numpy.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
    pists=numpy.angle(spot_fields)
    ints=numpy.abs(spot_fields)**2

    if algorithm=="CSWGS":
        weights=weights*(numpy.mean(numpy.sqrt(ints/intensities))/numpy.sqrt(ints/intensities))
        weights=weights/numpy.sum(weights)

        slm_total_field=numpy.sum((intensities*weights)[:,None]/(float(pupilCoords[0].shape[0]))*numpy.exp(1.0j*(slm_p_phase+pists[:,None])),axis=0)
        slm_total_phase=numpy.angle(slm_total_field)

        spot_fields=numpy.sum(pupilInt[None,:]*numpy.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
        pists=numpy.angle(spot_fields)
        ints=numpy.abs(spot_fields)**2
    
    outPhase=numpy.zeros(phaseRes,dtype="uint32")
    pupilPhase=numpy.zeros(res**2,dtype="uint32")
    pupilPhase[pupilCoords]=((slm_total_phase+numpy.pi)/(2*numpy.pi)*(2**16)).astype("uint32")
    outPhase[corner[0]:corner[0]+res,corner[1]:corner[1]+res]=pupilPhase.reshape((res,res))

    numpy.copyto(phase ,outPhase)
    numpy.copyto(performance ,numpy.asarray([numpy.sum(ints),1-(numpy.max(ints/intensities)-numpy.min(ints/intensities))/(numpy.max(ints/intensities)+numpy.min(ints/intensities))],dtype="float64"))


class hologramFft:
    def __init__(self,images,zs,pixsizes,superres,playerInst,sourceIndex,apertureIndex):
        self.images=images
        self.zs=zs
        self.pixsizes=pixsizes
        self.superres=superres
        self.playerInst=playerInst
        self.sourceIndex=sourceIndex
        self.apertureIndex=apertureIndex
        self.performance=numpy.zeros(2,dtype="float64")
        self.algorithm=None
        self.phase=numpy.zeros(playerInst.resolution,dtype="uint32")
        self.p=None
        for i in range(len(images)):
            if images[i].shape[0]*pixsizes[i]>2*playerInst.maxTilt(0.0,sourceIndex,apertureIndex) or images[i].shape[1]*pixsizes[i]>2*playerInst.maxTilt(0.0,sourceIndex,apertureIndex):
                raise ValueError("Image size too big for SLM resolution")

    def compute(self,algorithm,iterations=1,compression=1.0,cpuId=None):
        phaseRaw=RawArray("c",self.playerInst.resolution[0]*self.playerInst.resolution[1]*numpy.dtype("uint32").itemsize)
        self.phase=numpy.frombuffer(phaseRaw, dtype="uint32").reshape(self.playerInst.resolution)
        performanceRaw=RawArray("c",2*numpy.dtype("float64").itemsize)
        self.performance=numpy.frombuffer(performanceRaw, dtype="float64")
        lam=self.playerInst.sources[self.sourceIndex][0]*10**(-9)
        res=self.playerInst.apertures[self.apertureIndex][1]
        corner=self.playerInst.apertures[self.apertureIndex][2]
        pixSize=self.playerInst.pixSize*10**(-6)
        focal=self.playerInst.focal*10**(-3)
        pupilInt=numpy.copy(self.playerInst.sources[self.sourceIndex][1][corner[0]:corner[0]+res,corner[1]:corner[1]+res])
        self.p=Process(target=gsLoopFft,args=(phaseRaw,self.playerInst.resolution,performanceRaw,self.images,self.zs,self.pixsizes,self.superres,self.playerInst.abBases[self.playerInst.apertures[self.apertureIndex][3]][3],lam,corner,res,pixSize,focal,pupilInt,algorithm,iterations,compression,cpuId))
        self.p.daemon=True
        self.p.start()
        self.algorithm=algorithm
        
        print('aperture',res,'pixSize',pixSize,'source', self.playerInst.sources[self.sourceIndex][1].shape )
    def wait(self):
        if self.p is not None:
            self.p.join()


def gsLoopFft(phaseRaw,phaseRes,performanceRaw,images,zs,pixsizes,superres,abDataset,lam,corner,res,pixSize,focal,pupilInt,algorithm,iterations,compression,cpuId):
    p=psutil.Process()
    if cpuId is not None and cpuId<cpu_count():
        p.cpu_affinity([cpuId+2])
    else:
        p.cpu_affinity(range(2,cpu_count()))
    phase=numpy.frombuffer(phaseRaw, dtype="uint32").reshape(phaseRes)
    performance=numpy.frombuffer(performanceRaw, dtype="float64")
#----------------------------------------------------------------------------------

    holograms=[]
    for i in range(len(images)):
        if images[i].shape[0]!=images[i].shape[1]:
            if images[i].shape[0]<images[i].shape[1]:
                if (images[i].shape[1]-images[i].shape[0])%2==0:
                    padding=((int((images[i].shape[1]-images[i].shape[0])/2),(int((images[i].shape[1]-images[i].shape[0])/2))),(0,0))
                else:
                    padding=((int((images[i].shape[1]-images[i].shape[0])/2)+1,(int((images[i].shape[1]-images[i].shape[0])/2))),(0,0))
            else:
                if (images[i].shape[0]-images[i].shape[1])%2==0:
                    padding=((0,0),(int((images[i].shape[0]-images[i].shape[1])/2),(int((images[i].shape[0]-images[i].shape[1])/2))))
                else:
                    padding=((0,0),(int((images[i].shape[0]-images[i].shape[1])/2)+1,(int((images[i].shape[0]-images[i].shape[1])/2))))                
            image=numpy.pad(images[i], padding ,'constant', constant_values=0)
        else:
            image=numpy.copy(images[i])
        max_fov=lam*focal/(2*pixSize)
        fft_pix_size=2*max_fov/res #nyquist
        if superres[i]:#here
            scaling_factor=2*pixsizes[i]*10**(-6)/fft_pix_size
            # scaling_factor=pixsizes[i]*10**(-6)/fft_pix_size # LM
        else:
            scaling_factor=pixsizes[i]*10**(-6)/fft_pix_size
            print('here')
        image=numpy.asarray(PIL.Image.fromarray(image).resize((int(image.shape[0]*scaling_factor),int(image.shape[1]*scaling_factor)),PIL.Image.BILINEAR))
            
        if superres[i]:#here
            targetres=2*res
            gsPupilInt=numpy.pad(pupilInt/numpy.sum(pupilInt),int(res/2),'constant', constant_values=0)
        else:
            targetres=res
            gsPupilInt=pupilInt/numpy.sum(pupilInt)
            print('here')
        if image.shape[0]!=targetres:
            if (targetres-image.shape[0])%2==0:
                padding=int((targetres-image.shape[0])/2)
            else:
                padding=((int((targetres-image.shape[0])/2),int((targetres-image.shape[0])/2)+1),(int((targetres-image.shape[0])/2),int((targetres-image.shape[0])/2)+1))
            image=numpy.pad(image, padding ,'constant', constant_values=0)        
            
            
#----------------------------------------------------------------------------------
        image=image/numpy.sum(image)
        field=image*numpy.exp(1j*numpy.random.random(image.shape)*2*numpy.pi)
        pupilfield=numpy.exp(1j*numpy.angle(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(field)))))*gsPupilInt
        pupilfield=pupilfield/numpy.sum(pupilfield)
        if algorithm=="GS" or algorithm=="WGS":
            for n in range(iterations):
                imagefield=numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(pupilfield)))
                if algorithm=="WGS":
                    image[numpy.where(image!=0)]=image[numpy.where(image!=0)]*(numpy.mean(numpy.sqrt(numpy.abs(imagefield[numpy.where(image!=0)])**2))/numpy.sqrt(numpy.abs(imagefield[numpy.where(image!=0)])**2))
                    image=image/numpy.sum(image)
                imagefield=image*numpy.exp(1j*numpy.angle(imagefield))
                pupilfield=pupilfield=numpy.exp(1j*numpy.angle(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.fftshift(imagefield)))))*gsPupilInt
                pupilfield=pupilfield/numpy.sum(numpy.abs(pupilfield))

        holograms.append(numpy.angle(pupilfield[int((image.shape[0]-pupilInt.shape[0])/2):int((image.shape[0]-pupilInt.shape[0])/2+pupilInt.shape[0]),int((image.shape[1]-pupilInt.shape[1])/2):int((image.shape[1]-pupilInt.shape[1])/2+pupilInt.shape[1])]))

    xc,yc=numpy.meshgrid(numpy.linspace(-1.0,1.0,res),numpy.linspace(-1.0,1.0,res))
    xc=xc.flatten()
    yc=yc.flatten()
    pupilCoords=numpy.where(numpy.sqrt(xc**2+yc**2)<=1.0)
    xc=xc*res*pixSize/2.0
    yc=yc*res*pixSize/2.0

    if len(images)==1:     
        slm_total_phase=(holograms[0].flatten()[pupilCoords]+((numpy.pi*zs[0]*10**(-6))/(lam*(focal)**2)*(xc**2+yc**2))[pupilCoords])%(2*numpy.pi)
        outPhase=numpy.zeros(phaseRes,dtype="uint32")
        pupilPhase=numpy.zeros(res**2,dtype="uint32")
        pupilPhase[pupilCoords]=((slm_total_phase+numpy.pi)/(2*numpy.pi)*(2**16)).astype("uint32")
        outPhase[corner[0]:corner[0]+res,corner[1]:corner[1]+res]=pupilPhase.reshape((res,res))

        numpy.copyto(phase ,outPhase)
    else:        
        weights=numpy.ones(len(images))/float(len(images))
        pupilInt=pupilInt.flatten()
        pupilInt=pupilInt[pupilCoords]/numpy.sum(pupilInt[pupilCoords])
        pists=numpy.random.random(len(images))*2*numpy.pi
        
        slm_p_phase=numpy.zeros((len(images),pupilCoords[0].shape[0]),dtype="float32")
        for i in range(len(images)):                 
            slm_p_phase[i,:]=(holograms[i].flatten()[pupilCoords]+((numpy.pi*zs[i]*10**(-6))/(lam*(focal)**2)*(xc**2+yc**2))[pupilCoords])%(2*numpy.pi)



        if algorithm is not "RS":
            for i in range(iterations):

                slm_total_field=numpy.sum(weights[:,None]/(float(pupilCoords[0].shape[0]))*numpy.exp(1.0j*(slm_p_phase+pists[:,None])),axis=0)
                slm_total_phase=numpy.angle(slm_total_field)

                spot_fields=numpy.sum(pupilInt[None,:]*numpy.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
                pists=numpy.angle(spot_fields)
                
                
                if algorithm=="WGS":
                    ints=numpy.abs(spot_fields)**2
                    weights=weights*(numpy.mean(numpy.sqrt(ints))/numpy.sqrt(ints))
                    weights=weights/numpy.sum(weights)   

        

        slm_total_field=numpy.sum(weights[:,None]/(float(pupilCoords[0].shape[0]))*numpy.exp(1.0j*(slm_p_phase+pists[:,None])),axis=0)
        slm_total_phase=numpy.angle(slm_total_field)
        


        spot_fields=numpy.sum(pupilInt[None,:]*numpy.exp(1j*(slm_total_phase[None,:]-slm_p_phase)),axis=1)
        pists=numpy.angle(spot_fields)
        ints=numpy.abs(spot_fields)**2


        
        outPhase=numpy.zeros(phaseRes,dtype="uint32")
        pupilPhase=numpy.zeros(res**2,dtype="uint32")
        pupilPhase[pupilCoords]=((slm_total_phase+numpy.pi)/(2*numpy.pi)*(2**16)).astype("uint32")
        outPhase[corner[0]:corner[0]+res,corner[1]:corner[1]+res]=pupilPhase.reshape((res,res))

        numpy.copyto(phase ,outPhase)
##        numpy.copyto(performance ,numpy.asarray([numpy.sum(ints),1-(numpy.max(ints)-numpy.min(ints))/(numpy.max(ints)+numpy.min(ints))],dtype="float64"))
           


#---------------------------------------------------------------------------------------
##Player class

    

def put_array(surface, myarr):          # put array into surface
    bv = surface.get_view("0")
    bv.write(myarr.tostring())

class player:
    def __init__(self,pixSize,focal,frameTime,screenID=None,activeAreaCoords=None,apertures=[],sources=[]):
        self.screenID=screenID
        if self.screenID is not None:
            if activeAreaCoords is None:
                self.resolution=(screeninfo.get_monitors()[self.screenID].height,screeninfo.get_monitors()[self.screenID].width)
                self.position=(screeninfo.get_monitors()[self.screenID].x,screeninfo.get_monitors()[self.screenID].y)                
            else:
                self.resolution=(activeAreaCoords[2],activeAreaCoords[3])
                self.position=(screeninfo.get_monitors()[self.screenID].x+activeAreaCoords[0],screeninfo.get_monitors()[self.screenID].y+activeAreaCoords[1])
        else:
            if activeAreaCoords is None:
                self.resolution=(512,512)
                self.position=(0,0)
            else:
                self.resolution=(activeAreaCoords[2],activeAreaCoords[3])
                self.position=(activeAreaCoords[0],activeAreaCoords[1])
            
        self.pixSize=pixSize
        self.focal=focal
        self.frameTime=frameTime
        
        self.sequenceExposuresRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(0)
        self.sequenceRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape(0)
        self.sequenceStatusRaw=RawArray("c",2*numpy.dtype("uint32").itemsize)
        self.sequenceStatus=numpy.frombuffer(self.sequenceStatusRaw, dtype="uint32").reshape(2)

        self.isoAberrationRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.isoAberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        self.singleFrameRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.singleFrame=numpy.frombuffer(self.singleFrameRaw, dtype="uint32").reshape(self.resolution)
        self.p=None
       
        self.mode="singleFrame"
        self.working=False
        self.newFrame=True
        self.apertures=apertures #[fullframe coords,fftSquareSize,fftSquareCorner,aberrationsindex]
        self.currentAperture=0
        self.sources=sources #[wavelength,profile,LUT]
        self.currentSource=0
        self.abBases=[]#[base,resolution,order,bigarray]

    @classmethod
    def fromFile(cls,filename):
        loadFile=open(filename,"rb")
        playerInst=pickle.load(loadFile)
        loadFile.close()
        playerInst.sequenceExposuresRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        playerInst.sequenceExposures=numpy.frombuffer(playerInst.sequenceExposuresRaw, dtype="uint32").reshape(0)
        playerInst.sequenceRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        playerInst.sequence=numpy.frombuffer(playerInst.sequenceRaw, dtype="uint32").reshape(0)
        playerInst.sequenceStatusRaw=RawArray("c",2*numpy.dtype("uint32").itemsize)
        playerInst.sequenceStatus=numpy.frombuffer(playerInst.sequenceStatusRaw, dtype="uint32").reshape(2)

        playerInst.isoAberrationRaw=RawArray("c",playerInst.resolution[0]*playerInst.resolution[1]*numpy.dtype("uint32").itemsize)
        playerInst.isoAberration=numpy.frombuffer(playerInst.isoAberrationRaw, dtype="uint32").reshape(playerInst.resolution)
        playerInst.singleFrameRaw=RawArray("c",playerInst.resolution[0]*playerInst.resolution[1]*numpy.dtype("uint32").itemsize)
        playerInst.singleFrame=numpy.frombuffer(playerInst.singleFrameRaw, dtype="uint32").reshape(playerInst.resolution)
        playerInst.mode="singleFrame"

        return playerInst

    def maxTilt(self,z,sourceIndex,apertureIndex): #in um
        wl=self.sources[sourceIndex][0]
        print(self.apertures[apertureIndex][1],self. pixSize)
        return wl*10**(-3)*self.focal*(10**3)/(2*self.pixSize)-0.5*self.focal*(10**3)*z*self.apertures[apertureIndex][1]*self.pixSize**2
      


    def play(self):
        numpy.copyto(self.sequenceStatus,numpy.asarray([1,self.sequenceStatus[1]],dtype=("uint32")))

    def pause(self):
        numpy.copyto(self.sequenceStatus,numpy.asarray([0,self.sequenceStatus[1]],dtype=("uint32")))

    def stop(self):
        numpy.copyto(self.sequenceStatus,numpy.asarray([0,0],dtype=("uint32")))

    def setFrame(self,frame):
        numpy.copyto(self.sequenceStatus[1],numpy.asarray([self.sequenceStatus[0],frame],dtype=("uint32")))

    def prevFrame(self):
        if self.sequenceStatus[0]==0:
            numpy.copyto(self.sequenceStatus,numpy.asarray([self.sequenceStatus[0],self.sequenceStatus[1]-1]))

    def nextFrame(self):
        if self.sequenceStatus[0]==0:
            numpy.copyto(self.sequenceStatus,numpy.asarray([self.sequenceStatus[0],self.sequenceStatus[1]+1]))
    
    def setSequence(self,phasesSequence,exposuresSequence=None,abCoeffsSequence=None):
        if abCoeffsSequence is not None:
            for i in range(len(abCoeffsSequence)):
                if abCoeffsSequence[i] is not None:
                    phasesSequence[i]=phasesSequence[i]+(self.getAberrationPhase(abCoeffsSequence[i]))%(2**16)
                    
        self.sequenceExposuresRaw=RawArray("c",len(phasesSequence)*numpy.dtype("uint32").itemsize)
        self.sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(len(phasesSequence))
        self.sequenceRaw=RawArray("c",len(phasesSequence)*self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape((self.resolution[0],self.resolution[1],len(phasesSequence)))
        numpy.copyto(self.sequence,numpy.dstack(phasesSequence))
        if exposuresSequence is not None:
            numpy.copyto(self.sequenceExposures,numpy.asarray(exposuresSequence).astype("uint32"))
        else:
            numpy.copyto(self.sequenceExposures,numpy.asarray([self.frameTime]*len(phasesSequence),dtype=("uint32")))
        numpy.copyto(self.sequenceStatus,numpy.asarray([0,0],dtype=("uint32")))
        self.mode="sequence"
        self.stopLoop()
        self.startLoop()


    def addAberrationBase(self,base,res,order):
        newAbNeeded=True
        for ab in self.abBases:
            if ab[0]==base and ab[1]==res:
                newAbNeeded=False
                if order>ab[2]:
                    if base!="Hadamard":
                        ab[3]=numpy.concatenate((ab[3],generateAberrationDataset(res,base,order,ab[3].shape[1])),axis=1)
                    else:
                        ab[3]=generateAberrationDataset(res,base,order)
                index=self.abBases.index(ab)
        if newAbNeeded:
            self.abBases.append([base,res,order,generateAberrationDataset(res,base,order)])
            index=len(self.abBases)-1
        return index

    def getAberrationPhase(self,coeffs):
        abOut=numpy.zeros(self.resolution,dtype="float32")
        x0,x1=self.apertures[self.currentAperture][2]
        res=self.apertures[self.currentAperture][1]
        abOut[x0:x0+res,x1:x1+res]=getAberrationPhase(self.abBases[self.apertures[self.currentAperture][3]][3],self.sources[self.currentSource][0],coeffs)
        return abOut.astype("uint32")


    def setIsoAberration(self,coeffs):
        numpy.copyto(self.isoAberration,self.getAberrationPhase(coeffs))

    def setIsoAberrationPhase(self,aberrationPhase):
        if aberrationPhase.dtype=="float":
            aberrationPhase=(aberrationPhase/(2*numpy.pi)*(2**16)).astype("uint32")
            print(type(aberrationPhase[1,1]))
        numpy.copyto(self.isoAberration,aberrationPhase)

    def setPhase(self,phase):
        if phase.dtype=="float":
            phase=(phase/(2*numpy.pi)*(2**16)).astype("uint32")        
        numpy.copyto(self.singleFrame,phase)
        self.mode="singleFrame"
        self.stopLoop()
        self.startLoop()

    def setHologram(self,hologram):
        numpy.copyto(self.singleFrame,hologram.phase)
        self.mode="singleFrame"
        self.stopLoop()
        self.startLoop()
                
    def addSource(self,lam,sourceType,radius,x0,y0,LUT):
        xcoords,ycoords=numpy.meshgrid(numpy.linspace(-self.resolution[1]/2,self.resolution[1]/2,self.resolution[1]),numpy.linspace(-self.resolution[0]/2,self.resolution[0]/2,self.resolution[0]))
        if sourceType=="Round":
            profile=numpy.zeros(self.resolution)
            profile[numpy.where(numpy.sqrt((xcoords-x0)**2+(ycoords-y0)**2)<=radius/self.pixSize)]=1.0
        elif sourcetype=="Gaussian":
            profile=numpy.exp(-((xcoords-x0)**2+(ycoords-y0)**2)/((radius/self.pixSize)**2))
        self.sources.append([lam,profile,LUT])
        if len(self.sources)==1:
            self.setSource(0)

    def setSource(self,sourceIndex):
        restart=False
        if self.p is not None:
            restart=True
            self.stopLoop()
        self.currentSource=sourceIndex
        self.sequenceExposuresRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(0)
        self.sequenceRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape(0)
        self.sequenceStatusRaw=RawArray("c",2*numpy.dtype("uint32").itemsize)
        self.sequenceStatus=numpy.frombuffer(self.sequenceStatusRaw, dtype="uint32").reshape(2)

        self.isoAberrationRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.isoAberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        self.singleFrameRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.singleFrame=numpy.frombuffer(self.singleFrameRaw, dtype="uint32").reshape(self.resolution)
        self.mode="singleFrame"
        if restart:
            self.startLoop()
       

    def addAperture(self,radius,x0,y0,aberrations=None):
        xcoords,ycoords=numpy.meshgrid(numpy.linspace(-self.resolution[1]/2,self.resolution[1]/2,self.resolution[1]),numpy.linspace(-self.resolution[0]/2,self.resolution[0]/2,self.resolution[0]))
        pupilCoords=numpy.where(numpy.sqrt((xcoords-x0)**2+(ycoords-y0)**2)<=(radius/self.pixSize))
        fftSquareSize=min(numpy.max(pupilCoords[0])-numpy.min(pupilCoords[0]),numpy.max(pupilCoords[1])-numpy.min(pupilCoords[1]))+1
        fftSquareCorner=numpy.min(pupilCoords[0]),numpy.min(pupilCoords[1])
        if aberrations is not None:
            self.apertures.append([pupilCoords,fftSquareSize,fftSquareCorner,self.addAberrationBase(aberrations[0],fftSquareSize,aberrations[1])])
        else:
            self.apertures.append([pupilCoords,fftSquareSize,fftSquareCorner,None])
        if len(self.apertures)==1:
            self.setAperture(0)


    def setAperture(self,apertureIndex):
        restart=False
        if self.p is not None:
            restart=True
            self.stopLoop()
        self.currentAperture=apertureIndex
        self.sequenceExposuresRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(0)
        self.sequenceRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape(0)
        self.sequenceStatusRaw=RawArray("c",2*numpy.dtype("uint32").itemsize)
        self.sequenceStatus=numpy.frombuffer(self.sequenceStatusRaw, dtype="uint32").reshape(2)

        self.isoAberrationRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.isoAberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        self.singleFrameRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.singleFrame=numpy.frombuffer(self.singleFrameRaw, dtype="uint32").reshape(self.resolution)
        self.mode="singleFrame"
        if restart:
            self.startLoop()

    def stopLoop(self):
        if self.p is not None:
            self.p.terminate()
            self.p.join()


    def startLoop(self):
##        self.stopLoop()
        if self.mode=="singleFrame":
            self.p=Process(target=self.pyGameLoopSingleFrame,args=(self.resolution,self.position,self.sources[self.currentSource][2],self.frameTime))
            self.p.daemon=True
            self.p.start()
        if self.mode=="sequence":
            self.p=Process(target=self.pyGameLoopSequence,args=(self.resolution,self.position,self.sources[self.currentSource][2],self.frameTime))
            self.p.daemon=True
            self.p.start()

    def close(self):
        self.stopLoop()       


    def getApertureFftImageSize(self,index):
        return (self.apertures[index][1],self.apertures[index][1])

    def pyGameLoopSingleFrame(self,resolution,position,LUT,frameTime):
        p=psutil.Process()
        p.cpu_affinity([1])
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (position[0],position[1])
        pygame.init()
        windowSurface = pygame.display.set_mode((resolution[1],resolution[0]), pygame.NOFRAME)
        pygame.display.flip()
        screenImage=numpy.zeros(self.resolution,dtype="uint32")
        phase=numpy.frombuffer(self.singleFrameRaw, dtype="uint32").reshape(self.resolution)
        aberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        while 1:
            t=time.clock()
            pygame.event.pump()
            numpy.add(phase,aberration,out=screenImage)
            numpy.take(LUT,screenImage%(2**16),out=screenImage,mode="wrap")
            bv = windowSurface.get_view("0")
            bv.write(screenImage.tostring())
            pygame.display.flip()
            if time.clock()-t>float(frameTime)/1000.0:
                totalFrameTime=time.clock()-t
            else:
                while time.clock()-t<float(frameTime)/1000.0:
                    time.sleep(0.001)        

    def pyGameLoopSequence(self,resolution,position,LUT,frameTime):
        p=psutil.Process()
        p.cpu_affinity([1])
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (position[0],position[1])
        pygame.init()
        windowSurface = pygame.display.set_mode((resolution[1],resolution[0]), pygame.NOFRAME)
        pygame.display.flip()
        screenImage=numpy.zeros(self.resolution,dtype="uint32")
        aberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape(self.sequence.shape)
        sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(sequence.shape[2])
        sequenceStatus=numpy.frombuffer(self.sequenceStatusRaw, dtype="uint32").reshape(2)
        tFrame=time.clock()
        while 1:
            t=time.clock()
            pygame.event.pump()
            numpy.add(sequence[:,:,sequenceStatus[1]],aberration%(2**16),out=screenImage)
            numpy.take(LUT,screenImage,out=screenImage,mode="wrap")
            bv = windowSurface.get_view("0")
            bv.write(screenImage.tostring())
            pygame.display.flip()
            if sequenceStatus[0]==1 and time.clock()-tFrame>sequenceExposures[sequenceStatus[1]]/1000.0:
                newFrame=sequenceStatus[1]+1
                if newFrame>=sequence.shape[2]:
                    newFrame=0
                numpy.copyto(sequenceStatus,numpy.asarray([sequenceStatus[0],newFrame],dtype=("uint32")))
                tFrame=time.clock()
            if time.clock()-t>frameTime/1000.0:
                totalFrameTime=time.clock()-t
            else:
                while time.clock()-t<frameTime/1000.0:
                    time.sleep(0.001) 

    def saveToDisk(self,filename):
        self.stopLoop()
        self.sequenceExposuresRaw=None
        self.sequenceExposures=None
        self.sequenceRaw=None
        self.sequence=None
        self.sequenceStatusRaw=None
        self.sequenceStatus=None

        self.isoAberrationRaw=None
        self.isoAberration=None
        self.singleFrameRaw=None
        self.singleFrame=None
        self.mode="singleFrame"
        self.p=None
        saveFile=open(filename,"wb")
        pickle.dump(self,saveFile)
        saveFile.close()
        self.sequenceExposuresRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequenceExposures=numpy.frombuffer(self.sequenceExposuresRaw, dtype="uint32").reshape(0)
        self.sequenceRaw=RawArray("c",0*numpy.dtype("uint32").itemsize)
        self.sequence=numpy.frombuffer(self.sequenceRaw, dtype="uint32").reshape(0)
        self.sequenceStatusRaw=RawArray("c",2*numpy.dtype("uint32").itemsize)
        self.sequenceStatus=numpy.frombuffer(self.sequenceStatusRaw, dtype="uint32").reshape(2)

        self.isoAberrationRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.isoAberration=numpy.frombuffer(self.isoAberrationRaw, dtype="uint32").reshape(self.resolution)
        self.singleFrameRaw=RawArray("c",self.resolution[0]*self.resolution[1]*numpy.dtype("uint32").itemsize)
        self.singleFrame=numpy.frombuffer(self.singleFrameRaw, dtype="uint32").reshape(self.resolution)
        self.mode="singleFrame"        

            
if __name__=="__main__":
##    SLM=player.fromFile("test.slm")
    SLM=player(12.5,250,16,0)
##    LUT=numpy.linspace(0,2**16,2**16).astype("uint32")
    LUT=generateLinearLUT(0,140,1,1,1)
    SLM.addSource(488.0,"Round",3750,0.0,0.0,LUT)
    SLM.addAperture(3750.0,0.0,0.0,["Lukosz",7])
    SLM.startLoop()
    xc,yc=numpy.meshgrid(numpy.linspace(-1500.0,1500.0,5),numpy.linspace(-1500.0,1500.0,5))
    coords=numpy.zeros((25,3))
    coords[:,0]=xc.flatten()
    coords[:,1]=yc.flatten()
    abslist=[]
    for i in range(25):
        coeffs=numpy.zeros(28)
        coeffs[i+3]=1.0
        abslist.append(coeffs)
        


    holo=hologram(coords,numpy.ones(25),abslist,SLM,0,0)
    holo.compute("CSWGS",20,0.2,cpuId=1)

    holo.wait()


    SLM.setPhase(holo.phase)

    for i in range(25):
        print(i)
        coeffs=numpy.zeros(28)
        coeffs[i+3]=-1.0
        SLM.setIsoAberration(coeffs)
        time.sleep(1.0)

    SLM.close()

##    
##    
##
##    holoSequence=[]
##    phaseSequence=[]
##    for i in range(20):
####        images=[]
####        for l in range(3):
####            image=numpy.zeros((512,512))
####            for j in numpy.linspace(128,128+256,5).astype("uint32"):
####                for k in numpy.linspace(128,128+256,5).astype("uint32"):
####                    image[j,k]=1.0
####            images.append(image)
####
####        holo=hologramFft([image],[-50.0+5.0*i,5.0*i,50+5.0*i],SLM,0,0)
##        xc,yc=numpy.meshgrid(numpy.linspace(-300.0,300.0,5),numpy.linspace(-300.0,300.0,5))
##        coords=numpy.zeros((25,3))
##        coords[:,0]=xc.flatten()
##        coords[:,1]=yc.flatten()
##        coords[:,2]=(i-10)*500.0
##
##        holo=hologram(coords,None,SLM,0,0)
##        if i/2==0:
##            holo.compute("CSWGS",10,0.2,cpuId=i%2+2)
##        else:
##            holoSequence[i-2].wait()
##            holo.compute("CSWGS",10,0.2,cpuId=i%2+2)
##        holoSequence.append(holo)
##
##    for holo in holoSequence:
##        holo.wait()
##        phaseSequence.append(numpy.copy(holo.phase))
##    c=0
##
##    SLM.setSequence(phaseSequence,[100]*20)
##
##
##    SLM.play()
##    print "Playing"

##    time.sleep(10)
##
##    SLM.close()
    


##    for i in range(45):
##        print i
##        coeffs=numpy.zeros(45)
##        coeffs[i]=0.488*3
##        p.setIsoAberration(coeffs)
##        time.sleep(2.0)

