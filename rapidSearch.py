import multiprocessing
import glob,h5py
import pickle
import cv2
import numpy as np 


def load_files(files):
    h5fs={}
    for i ,f in enumerate(sorted(files)):
        # print f
        #h5fs['h5f_'+str(i)]=h5py.File(f,'r')
        h5fs[i]=h5py.File(f,'r')


    return h5fs   
def transform(L,h5fs):
    # print L
    #for i in range(len(L)):
        #print h5fs[i]['centroids'][L[i]]
        #L[i]=h5fs[i]['centroids'][L[i]]
    L=np.concatenate([h5fs[i]['centroids'][L[i]] for i in range(8)])    
    #return np.concatenate([f for f in L]) 
    # print L.shape,L
    return L
def coarseQuantization(Q,feats):
    dists=((Q-feats)**2).sum(axis=1)
    idx=np.argsort(dists)
    Q=Q-feats[idx[0]]
    return idx,Q
   

def compaireResult(Q,idx,Pq,h5fs):
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




if __name__=='__main__':
    files=glob.glob('feats/*')
    h5fs=load_files(files)
    #imgPath='data/595.jpg'
    imgPath='canny.JPEG'
    Q=testData(imgPath)
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
    i=0;count=0
    
    while(i<Q.shape[0]*0.5 and count<=7):
        idx,q=coarseQuantization(Q[i],feats)
        flag=compaireResult(q,idx,Pq,h5fs)
        #print flag
        if flag==1:
            count+=1
        i+=1    
    print count    
    
