#!/usr/bin/env python

# This function computes features by using pretrained CNN model.
# The features computed will be used as an input for visual attention model.

import  argparse
import  sys
import  numpy        as np
import  h5py
import  json
import  utils.config as      config
from    utils.helper import  *
from    src.model    import  *
import  tensorflow   as      tf
from    server_sid       import  client_generator


config.gpu_fraction = 0.6

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('--dataset',  type=str, default="2016-01-30--11-24-51", help='Dataset/video clip name')
  parser.add_argument('--fpath',    type=str, default="D:/DR/Dataset/comma-dataset/comma_dataset", help='Dataset local path')
  parser.add_argument('--model',    type=str, default="D:/DR/Dataset/Results/Train/model-30000.ckpt", help='Model path')
  args = parser.parse_args()

  # Open a tensorflow session
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_fraction)
  sess        = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  # Create a CNNs (NVIDIA) model
  NVIDIA_model = NVIDIA_CNNs(sess)

  # Preprocessor
  pre_processor = PreProcessor()

  # saver
  saver = tf.train.Saver(max_to_keep=40)
  if args.model is not None:
    saver.restore(sess, args.model)
    print ("Loaded the pretrained model: %s"%(args.model))

  fpath   = args.fpath
  dataset = args.dataset

  log = h5py.File(fpath+"/log/"    +dataset+".h5", "r")
  cam = h5py.File(fpath+"/camera/" +dataset+".h5", "r")

  print (log.keys())

  feats = []
  for i in range( 0, cam['X'].shape[0] ):
    if i%100 == 0:
      print ("%.2f seconds elapsed" % ( i/100.0 ))

    # load data  
    img           = cam['X'][i]
    angle_steers  = log['steering_angle'][i]
    speed_ms      = log['speed'][i]

    # Preprocessing
    X, curvature, angle = pre_processor.process(sess, img[None,None,:,:,:], [[[angle_steers]]], [[[speed_ms]]])

    # feats
    feat          = NVIDIA_model.extractFeats(sess, X) 
    feat          = np.squeeze( np.array(feat) )
    feat          = feat.transpose( 2,0,1 )

    # accumulation
    feats.append(feat)

  # create new dataset
  f     = h5py.File(fpath+"/features/%s.h5"%(dataset), "w")
  dset  = f.create_dataset("/X", data=feats, chunks=(1024,64,10,20))
  








