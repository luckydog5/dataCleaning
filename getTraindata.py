import h5py,os,glob
import cv2
import numpy as np 
import pickle
import multiprocessing
from sklearn.decomposition import PCA
import skimage.io as io
def extrac_sift(img):
    sift=cv2.xfeatures2d.SIFT_create()
    kp=sift.detect(img,None)
    descriptor=sift.compute(img,kp)
    des=np.concatenate([[i] for i in descriptor[1]])
    #print des.shape
    return des
def split_data(alist,wanted_parts=2):
    length=len(alist)
    return [alist[i*length//wanted_parts:(i+1)*length//wanted_parts] for i in range(wanted_parts)]
def extract_task(path_images,out):
    num_images=len(path_images)
    #h5f=h5py.File(out,'wb')
    features=[]
    image_names=[]
    for i,path in enumerate(path_images):
        #print "%d(%d), %s"%((i+1),num_images,os.path.basename(path))
        #d=opencv_format_img_for_vgg(path,True)
        #feat=extract_fc_features(net,layer,d)
        img=cv2.imread(path,1)
        feat=extrac_sift(img)
        print feat.shape
        features.append(np.array(feat))
        image_names.append(os.path.basename(path))
    features=np.array(features)
    ff=open(out,'wb')
    pickle.dump(features,ff)
    ff.close()
    """
    h5f['feats']=features
    h5f['names']=image_names
    h5f.close()
    """
    print "process  task has finished...."

if __name__=='__main__':
   
    dir_images='/home/mysj/caffe_file/klboat/*'
    pool=multiprocessing.Pool(processes=2)
    out_files=[]
    out='/home/mysj/caffe_file/feats'
    path_images=[os.path.join(dir_images,f)for f in sorted(glob.glob(dir_images))]
    for i in xrange(2):
        out_files.append(os.path.join(out,str(i)+'.pickle'))
    blocks=split_data(path_images,wanted_parts=2) 
    for i in xrange(0,2):
        pool.apply_async(extract_task,args=(blocks[i],out_files[i])) 
    pool.close()
    pool.join()
    """
    dir_images='/home/mysj/caffe_file/klboat/*'
    path_images=[os.path.join(dir_images,f)for f in sorted(glob.glob(dir_images))]
    print path_images[9]
    img=cv2.imread(path_images[9],1)
    cv2.imshow('9',img)
    cv2.waitKey(0)
    """