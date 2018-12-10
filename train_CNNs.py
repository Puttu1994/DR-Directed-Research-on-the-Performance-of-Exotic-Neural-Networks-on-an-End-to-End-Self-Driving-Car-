#!/usr/bin/env python
"""
Steering angle prediction model (NVIDIA)

how to run:
python server.py --batch 200 --port 5557 --time 1
python server.py --batch 200 --validation --port 5556 --time 1
python train.py --port 5557 --val_port 5556
"""

import tensorflow as tf
import  os
import  sys
import  argparse
import  json
import  numpy as np
from    server_sid import  client_generator
import  utils.config as config
from    utils.helper import  *
from    src.model    import  *

txtpath = "D:\DR\Dataset\Results\Train"


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host',     type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port',     type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--log_path', type=str, default='./saved/log/')
  args = parser.parse_args()

  # Open a tensorflow session
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth = True)

  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement = False))

  # Create a CNNs (NVIDIA) model
  NVIDIA_model = NVIDIA_CNNs(sess)

  # Preprocessor
  pre_processor = PreProcessor()

  # Summary
  tf.summary.scalar('batch_loss', NVIDIA_model.l1)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  summary_op = tf.summary.merge_all()

  # saver
  saver = tf.train.Saver(max_to_keep=40)
  if config.pretrained_model_path is not None:
    saver.restore(sess, config.pretrained_model_path)
    print ("Loaded the pretrained model: %s"%(config.pretrained_model_path))

  # Train over the dataset
  data_train  = client_generator(hwm=20, host="localhost", port=args.port)
  print (data_train)
  data_val    = client_generator(hwm=20, host="localhost", port=args.val_port)

  # Summary writer
  summary_writer = tf.summary.FileWriter(args.log_path, graph=tf.get_default_graph())
  txt = open(txtpath+"\\"+"output.txt", 'w')

  for i in range(config.epochsize):
    # Load new dataset
    X_batch, angle_batch, speed_batch = next(data_train)

    # Preprocessing
    Xprep_batch, curvatures, angles = pre_processor.process(sess, X_batch, angle_batch, speed_batch)

    # Training
    if config.use_curvature:
      l1loss, _ = NVIDIA_model.process(sess, Xprep_batch, curvatures)
    else:
      l1loss, _ = NVIDIA_model.process(sess, Xprep_batch, angles)
    
      
    if (config.skipvalidate is not True) and (i%config.val_steps==0):
      X_val, angle_val, speed_val           = next(data_val)
      Xprep_val, curvatures_val, angles_val = pre_processor.process(sess, X_val, angle_val, speed_val)

      if config.use_curvature:
        l1loss_val, y_pred                    = NVIDIA_model.validate(sess, Xprep_val, curvatures_val)
      else:
        l1loss_val, y_pred                    = NVIDIA_model.validate(sess, Xprep_val, angles_val)
      
      
      
      txt.write("\rStep {}, train loss: {}, val loss: {}\n".format( i, l1loss, l1loss_val))
      # display
      print ("\n********** Iteration %i ************" % i)
      print("\rStep {}, train loss: {} val loss: {}".format( i, l1loss, l1loss_val))
      
      sys.stdout.flush()
      
      # summary op
      feed_dict = {NVIDIA_model.x: Xprep_batch, NVIDIA_model.y_: curvatures}
      summary = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary, 2*i)

      feed_dict = {NVIDIA_model.x: Xprep_val, NVIDIA_model.y_: curvatures_val}
      summary = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary, 2*i+1)

    elif (i%config.val_steps==0):
      # display
      print ("\n********** Iteration %i ************" % i)
      print("\rStep {}, train loss: {}".format( i, l1loss))
      sys.stdout.flush()
    
    if i%config.save_steps==0:
      checkpoint_path = os.path.join( config.model_path, "model-%d.ckpt"%i )
      filename        = saver.save(sess, checkpoint_path)
      print("Model saved in file: %s" % filename)


  # End of code

