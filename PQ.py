"""
Update 2018.9.17
Major contents are productQuantization.
Three functions:
coarseQuantization()
computeResidual()
invertfileList()

Update 2018.9.26
loadfile()
load_files()
invertfile has been done
"""

import glob
import pickle,h5py,os
import numpy as np 
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
def loadfile(datafile):
    datafiles=[]
    for f in datafile:
        ff=open(f,'rb')
        fff=pickle.load(ff)
        data=np.concatenate([i for i in fff])
        datafiles.append(data)
    return datafiles,np.concatenate([item for item in datafiles])

def load_files(files):
    h5fs={}
    for i ,f in enumerate(sorted(files)):
        #print f
        h5fs['h5f_'+str(i)]=h5py.File(f,'r')

    #feats=np.concatenate([value['feats']for key,value in h5fs.items()])
    #names=np.concatenate([value['names']for key,value in h5fs.items()])
    #labels=h5fs['h5f_0']['labels']
    
    #return (feats,names)
    return h5fs    
def split_data(DB,wantedparts,coarse=None):
    if coarse!=None:
        for i in range(0,s.shape[0]): 
            label=coarse['labels'][i] 
            #print label  
            s[i]=s[i]-coarse['centroids'][label]
    length=DB.shape[1]
    print DB.shape[1]
    return [DB[:,i*length//wantedparts:(i+1)*length//wantedparts]for i in xrange(wantedparts)]


def ccluster(n,batchs,db,out):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n,max_iter=1000, batch_size=batchs,
                      n_init=10, max_no_improvement=10, verbose=0).fit(db)
    label=mbk.labels_
    centroid=mbk.cluster_centers_
    h5f=h5py.File(out,'w')
    h5f['labels']=label
    h5f['centroids']=centroid
    h5f.close()
    print 'hello'
    #return label,centroid

def invertFileList(coarse,h5fs):
    Pq={}
    print coarse['labels'].shape[0]
    lenn=coarse['labels'].shape[0]
    #labell=np.concatenate([[value['labels']] for key,value in h5fs.items()])
    labell=np.concatenate([[h5fs['h5f_'+str(i)]['labels']] for i in range(8)])
    for i in range(lenn):
        if Pq.has_key(coarse['labels'][i]):
            Pq[coarse['labels'][i]].append(labell[:,i] )
        else:
            Pq[coarse['labels'][i]]=[labell[:,i]]  

    print len(Pq) 
    print Pq[0][0] 






if __name__=='__main__':
   
    
    path_data='data/*.pickle'
    datafile=glob.glob(path_data)
    datafiles,s=loadfile(datafile)
    #print len(datafiles),datafiles[1].shape,datafiles[0].shape
    #print s.shape,len(s)
    ####################################################################
    coarse=h5py.File('data/coarse.h5','r')   # get coarse centroid and label
    blocks=split_data(s,8,coarse=coarse)
    #print len(blocks), blocks[1].shape
    ###################################################################
    out='/home/mysj/caffe_file/feats/'
    out_files=[]
    #coarseOut=os.path.join(out,'coarse.h5')
    #ccluster(1024,500,s,coarseOut)

    files=glob.glob('feats/*.h5')
    h5fs=load_files(files)
    #print h5fs.keys()
    
    invertFileList(coarse,h5fs)



    """
        this is about 8 processes computing k-menas
    """


    """
    for i in xrange(8):
        out_files.append(os.path.join(out,str(i)+'.h5'))
    pool=multiprocessing.Pool(processes=8)
    for i in xrange(8):
        pool.apply_async(ccluster,args=(256,500,blocks[i],out_files[i]))
    pool.close()
    pool.join()  
    """
  
  


