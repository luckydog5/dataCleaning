"""
Update 2018.10.10
merge function coarseQuantization() and compaireResult,in addition,assign images to different processes 
"""
import multiprocessing
import glob,h5py
import pickle
import cv2
import numpy as np 
import time 
from multiprocessing import Process
from copy import deepcopy


def split_data(imgPath,wantedparts=8):
    length=len(imgPath)
    return [imgPath[i*length // wantedparts: (i+1)*length // wantedparts]for i in range(wantedparts)]


def load_files(files):
    h5fs={}
    for i ,f in enumerate(sorted(files)):
        # print f
        #h5fs['h5f_'+str(i)]=h5py.File(f,'r')
        h5fs[i]=h5py.File(f,'r')
    return h5fs   
"""
define h5fs as global variables
"""    
files=glob.glob('feats/*')
h5fs=load_files(files)    
def transform(L,h5fs):
    # print L
    #for i in range(len(L)):
        #print h5fs[i]['centroids'][L[i]]
        #L[i]=h5fs[i]['centroids'][L[i]]
    L=np.concatenate([h5fs[i]['centroids'][L[i]] for i in range(8)])    
    #return np.concatenate([f for f in L]) 
    # print L.shape,L
    return L
"""    
def coarseQuantization(Q,feats):
    dists=((Q-feats)**2).sum(axis=1)
    idx=np.argsort(dists)
    Q=Q-feats[idx[0]]
    return idx,Q
"""   

def compaireResult(Q,feats,Pq):

    dist=((Q-feats)**2).sum(axis=1)
    idx=np.argsort(dist)
    Q=Q-feats[idx[0]]


    subdata=Pq[idx[0]]
    # print subdata
    LL=[]
    for i in range(len(subdata)):
        L=subdata[i]
        L=transform(L,h5fs)
        LL.append(L)
    LLL=np.concatenate([[f] for f in LL]) 
    dists=((Q-LLL)**2).sum(axis=1)
    idx=np.argsort(dists)
    dists=dists[idx]   
    if dists[0]/dists[1]<=0.6:
        flag=1
    else:
        flag=0
    return flag

def testData(imgPath):
    img=cv2.imread(imgPath,1)
    img=cv2.resize(img,(384,256))
    sift=cv2.xfeatures2d.SIFT_create()
    kp=sift.detect(img,None)
    des=sift.compute(img,kp)
    siftData=des[1]
    #Q=siftData[0]                      # test one sift
    return siftData

def multiTask(pathImages,feats,Pq):
    for f in pathImages:

        Q=testData(f)
        i=0;count=0
    
        while(i<Q.shape[0] and count<=7):
            #idx,q=coarseQuantization(Q[i],feats)
            flag=compaireResult(Q[i],feats,Pq)
        #print flag
            if flag==1:
                count+=1
            i+=1    
        print count  
     


if __name__=='__main__':
    #multiprocessing.freeze_support()
    
    #imgPath='data/595.jpg'
    #imgPath='canny.JPEG'
    imgPath=glob.glob('data/*.jpg')
    
    f=h5py.File('data/coarse.h5','r')
    feats=f['centroids']
    

    ff=open('data/invertfile.pickle','rb')
    Pq=pickle.load(ff)
    """
    for i in range(Q.shape[0]):
        idx,q=coarseQuantization(Q[i],feats)
        # trans=transform(Pq[idx][0],h5fs)
    
        flag=compaireResult(q,idx,Pq,h5fs)

    """
    blocks=split_data(imgPath,8)
    """
    t0=time.time()
    
    p1=Process(target=multiTask,args=(blocks[0],feats,Pq))
    p2=Process(target=multiTask,args=(blocks[1],feats,Pq))
    p3=Process(target=multiTask,args=(blocks[2],feats,Pq))
    p4=Process(target=multiTask,args=(blocks[3],feats,Pq))
    p5=Process(target=multiTask,args=(blocks[4],feats,Pq))
    p6=Process(target=multiTask,args=(blocks[5],feats,Pq))
    p7=Process(target=multiTask,args=(blocks[6],feats,Pq))
    p8=Process(target=multiTask,args=(blocks[7],feats,Pq))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    
    t1=time.time()
    print t1-t0
    """
    """
    multiTask(blocks[0],feats,Pq)
    multiTask(blocks[1],feats,Pq)
    t2=time.time()
    print t2-t1
    """



    
    
    
    pool=multiprocessing.Pool(processes=8)
    for i in xrange(0,8):
        pool.apply_async(multiTask,args=(blocks[i],feats,Pq))
    pool.close()
    pool.join()   
    
   


    """
    t0=time.time()
    for f in imgPath:

        Q=testData(f)
        i=0;count=0
    
        while(i<Q.shape[0] and count<=7):
            #idx,q=coarseQuantization(Q[i],feats)
            flag=compaireResult(Q[i],feats,Pq,h5fs)
        #print flag
            if flag==1:
                count+=1
            i+=1    
        print count    
    print time.time()-t0
    """
