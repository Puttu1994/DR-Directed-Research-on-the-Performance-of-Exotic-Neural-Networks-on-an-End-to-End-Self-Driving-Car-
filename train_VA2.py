#!/usr/bin/env python
"""
requires:
  tensorflow v.0.12
Steering angle prediction model (NVIDIA)
python server.py --port 5557 --time 20
python server.py --validation --port 5556 --time 20
python train_VA.py
"""

import  os
import  sys
import  argparse
import  json
import  numpy        as np
from    server       import  client_generator
import  utils.config as      config
from    utils.helper import  *
from    src.model    import  *


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host',     type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port',     type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--log_path', type=str, default='./saved/log/')
  args = parser.parse_args()

  # Create a Visual Attention model
  VA_model  = VA(alpha_c=0.0) 
  loss      = VA_model.build_model()
  # tf.get_variable_scope().reuse_variables()
  with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
    tf.get_variable_scope().reuse == True


  # Exponential learning rate decaying
  global_step           = tf.Variable(0, trainable=False)
  starter_learning_rate = config.lr
  learning_rate         = tf.train.exponential_decay(starter_learning_rate, global_step,
                                         1000, 0.96, staircase=True)

  # train op
  with tf.name_scope('optimizer'):
    optimizer       = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads           = tf.gradients(loss, tf.trainable_variables())
    grads_and_vars  = list(zip(grads, tf.trainable_variables()))
    train_op        = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

  # Summary
  tf.summary.scalar('batch_loss', loss)
  tf.summary.scalar('learning_rate', learning_rate)
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  for grad, var in grads_and_vars:
    tf.summary.histogram(var.op.name+'/gradient', grad)

  summary_op = tf.summary.merge_all()

  # Preprocessor
  pre_processor = PreProcessor_VA()

  # Open a tensorflow session
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  sess = tf.InteractiveSession(config=tfconfig)
  tf.global_variables_initializer().run()

  # saver
  saver = tf.train.Saver(max_to_keep=40)
  if config.pretrained_model_path is not None:
    saver.restore(sess, config.pretrained_model_path)
    print ("Loaded the pretrained model: %s"%(config.pretrained_model_path))

  # Summary writer
  summary_writer = tf.summary.FileWriter(args.log_path, graph=tf.get_default_graph())

  # Train over the dataset
  data_train  = client_generator(hwm=20, host="localhost", port=args.port)
  data_val    = client_generator(hwm=20, host="localhost", port=args.val_port)

  curr_loss = 0
  for i in range(config.epochsize):
    feats_batch, angle_batch, speed_batch = next(data_train)

    # Preprocessing
    feats, curvatures, angles = pre_processor.process(sess, feats_batch, angle_batch, speed_batch)

    # Training
    if config.use_curvature:
      feed_dict = {VA_model.features: feats, VA_model.y: curvatures}
    else:
      feed_dict = {VA_model.features: feats, VA_model.y: angles}

    _, l1loss      = sess.run([train_op, loss], feed_dict)

    if (config.skipvalidate is not True) and (i%config.val_steps==0):
      summary = sess.run(summary_op, feed_dict)
      summary_writer.add_summary(summary, i)

      feats_val, angle_val, speed_val         = next(data_val)
      Xprep_val, curvatures_val, angles_val   = pre_processor.process(sess, feats_val, angle_val, speed_val)

      if config.use_curvature:
        feed_dict = {VA_model.features: Xprep_val, VA_model.y: curvatures_val}
      else:
        feed_dict = {VA_model.features: Xprep_val, VA_model.y: angles_val}

      l1loss_val = sess.run([loss], feed_dict)

      # display
      print ("\n********** Iteration %i ************" % i)
      print("\rStep {}, train loss: {} val loss: {}".format( i, l1loss, l1loss_val))
      sys.stdout.flush()

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
