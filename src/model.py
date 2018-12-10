import tensorflow           as tf
from   utils.config         import *
import numpy                as np
from    utils.helper import  *
from src.alex import *
import capslayer as cl

class NVIDIA_CNNs(object):
  def __init__(self, sess):
    self.sess = sess

    # self.x   = tf.placeholder(tf.float32,
    #   shape=[None, int(config.imgRow/config.resizeFactor), int(config.imgCol/config.resizeFactor), config.imgCh])
    self.x   = tf.placeholder(tf.float32,
      shape=[None, int(config.imgRow/config.resizeFactor), int(config.imgCol/config.resizeFactor), config.imgCh])
    self.y_  = tf.placeholder(tf.float32, shape=[None, 1])
    self.create_net()

  def create_net(self):
    # # conv1: 24 filters / 5x5 kernel / 2x2 stride
    # conv1 = tf.contrib.layers.conv2d(self.x,24, 5, 2, activation_fn = tf.nn.relu)
    # # conv2: 36 filters / 5x5 kernel / 2x2 stride
    # conv2 = tf.contrib.layers.conv2d(conv1, 36, 5, 2, activation_fn = tf.nn.relu)
    # # conv1: 48 filters / 5x5 kernel / 2x2 stride
    # conv3 = tf.contrib.layers.conv2d(conv2, 48, 5, 2, activation_fn = tf.nn.relu)
    # # conv1: 24 filters / 3x3 kernel / 1x1 stride
    # conv4 = tf.contrib.layers.conv2d(conv3, 64, 3, 1, activation_fn = tf.nn.relu)
    # # conv1: 24 filters / 3x3 kernel / 1x1 stride
    # conv5 = tf.contrib.layers.conv2d(conv4, 64, 3, 1, activation_fn = tf.nn.relu)

    # # To extract features
    # self.conv5 = conv5

    # # fully connected layers
    # flattened = tf.contrib.layers.flatten(conv5)
    # fc1       = tf.contrib.layers.fully_connected(flattened,  1164, activation_fn = tf.nn.relu)
    # fc2       = tf.contrib.layers.fully_connected(fc1,        100,  activation_fn = tf.nn.relu)
    # fc3       = tf.contrib.layers.fully_connected(fc2,        50,   activation_fn = tf.nn.relu)
    # fc4       = tf.contrib.layers.fully_connected(fc3,        10,   activation_fn = tf.nn.relu)
    # self.y    = tf.contrib.layers.fully_connected(fc4,        1,    activation_fn = None)


    #Caps Net Testing
    # conv1: 24 filters / 5x5 kernel / 2x2 stride
    conv1 = tf.layers.conv2d(self.x,24, 9, 2, padding="VALID", activation = tf.nn.relu)
    conv1 = tf.nn.dropout(conv1, 0.8)
    # conv2: 36 filters / 5x5 kernel / 2x2 stride
    conv2 = tf.layers.conv2d(conv1, 36, 5, 2, activation = tf.nn.relu)
    conv2 = tf.nn.dropout(conv2, 0.8)
    # conv1: 48 filters / 5x5 kernel / 2x2 stride
    conv3 = tf.layers.conv2d(conv2, 48, 5, 2, activation = tf.nn.relu)
    conv3 = tf.nn.dropout(conv3, 0.8)
    # conv1: 24 filters / 3x3 kernel / 1x1 stride
    conv4 = tf.layers.conv2d(conv3, 64, 3, 1, activation = tf.nn.relu)
    conv4 = tf.nn.dropout(conv4, 0.8)
    # conv1: 24 filters / 3x3 kernel / 1x1 stride
    conv5 = tf.layers.conv2d(conv4, 64, 3, 1, activation = tf.nn.relu)
    conv5 = tf.nn.dropout(conv5, 0.8)

    # conv3 = tf.contrib.layers.conv2d(conv2,48,5,2,activation_fn=tf.nn.relu)
    conv_caps1, conv_caps1_activation = cl.layers.primaryCaps(conv1,64,9,2,[8,1],method='norm')
    # conv_caps1_activation = tf.nn.dropout(conv_caps1_activation, 0.4)
    # tf.assign(conv_caps1[1:4],conv_caps1_activation)

    conv_caps2, conv_caps2_activation = cl.layers.conv2d(conv_caps1, conv_caps1_activation, 32, [8,1], 9, 2,padding = "valid", routing_method = "EMRouting")
    # print ("conv2",conv_caps2)
    # print ("conv2a",conv_caps2_activation)
    # conv_caps3, conv_caps3_activation = cl.layers.conv2d(conv_caps2, conv_caps2_activation, 64, [4,1], 5, 2, padding="valid", name = "con22d", routing_method = "EMRouting" )

    # conv_caps4, conv_caps4_activation = cl.layers.conv2d(conv_caps3, conv_caps3_activation, 64, [2,1], 2, 1, padding="valid", name = "con23d", routing_method = "EMRouting" )

    # conv_caps3, conv_caps3_activation = cl.layers.primaryCaps(conv_caps1_activation,128,5,2,[32,1],method='norm')
    
    num_inputs = np.prod(cl.shape(conv_caps2)[1:4])
    conv_caps2 = tf.reshape(conv_caps2, shape=[-1, num_inputs, 8, 1])
    conv_caps2_activation = tf.reshape(conv_caps2_activation, shape=[-1, num_inputs])
    poses1, probs1 = cl.layers.dense(conv_caps2,
                                                     conv_caps2_activation,
                                                     num_outputs=1024,
                                                     out_caps_dims=[1, 1],
                                                     routing_method="EMRouting")

    # num_inputs = np.prod(cl.shape(poses1)[1:4])
    # poses1 = tf.reshape(poses1, shape=[-1, num_inputs, 16, 1])
    # probs1 = tf.reshape(probs1, shape=[-1, num_inputs])
    # poses2, probs2 = cl.layers.dense(poses1,
    #                                                  probs1,
    #                                                  num_outputs=1,
    #                                                  out_caps_dims=[1, 1],
    #                                                  routing_method="EMRouting", name = "dense2")

    
    # num_inputs = np.prod(cl.shape(poses2)[1:4])
    # poses2 = tf.reshape(poses2, shape=[-1, num_inputs, 4, 1])
    # probs2 = tf.reshape(probs2, shape=[-1, num_inputs])
    # poses3, probs3 = cl.layers.dense(poses2,
    #                                                  probs2,
    #                                                  num_outputs=1,
    #                                                  out_caps_dims=[1, 1],
    #                                                  routing_method="EMRouting", name = "dense3")


    # num_inputs = np.prod(cl.shape(poses3)[1:4])
    # poses3 = tf.reshape(poses3, shape=[-1, num_inputs, 2, 1])
    # probs3 = tf.reshape(probs3, shape=[-1, num_inputs])
    # poses4, probs4 = cl.layers.dense(poses3,
    #                                                  probs3,
    #                                                  num_outputs=1,
    #                                                  out_caps_dims=[4, 1],
    #                                                  routing_method="EMRouting", name = "dense4")

    # num_inputs = np.prod(cl.shape(poses1)[1:4])
    # poses1 = tf.reshape(poses1, shape=[-1, num_inputs, num_inputs, 1])
    # probs1 = tf.reshape(probs1, shape=[-1, num_inputs])
    # poses2, probs2 = cl.layers.dense(poses1,
    #                                                  probs1,
    #                                                  num_outputs=1,
    #                                                  out_caps_dims=[4, 1],
    #                                                  routing_method="EMRouting")

    # print("O\P",probs2)
    self.y = (probs1)
    # self.y = tf.argmax(cl.softmax(probs4,axis=1), axis = 1)
    #Caps Net test end

    #Alex Net Testing
    #   # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    # conv1 = conv(self.x, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    # norm1 = lrn(conv1, 2, 1e-05, 0.75, name = 'norm1')
    # # pool1 = max_pool(norm1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')

    # # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
    # conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
    # norm2 = lrn(conv2, 2, 1e-05, 0.75, name = 'norm2')
    # # pool2 = max_pool(norm2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')

    # # 3rd Layer: Conv (w ReLu)
    # conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')

    # # 4th Layer: Conv (w ReLu) splitted into two groups
    # conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

    # # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    # conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
    # # pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

    # # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    # flattened = tf.reshape(conv5, [-1, 6*6*256])
    # fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    # # dropout6 = dropout(fc6, self.KEEP_PROB)

    # # 7th Layer: FC (w ReLu) -> Dropout
    # fc7 = fc(fc6, 4096, 1164, relu = True, name = 'fc7')
    # fc8 = fc(fc7, 1164, 100, relu = True,name = 'fc8')
    # fc9 = fc(fc8, 100, 50, relu = True,name = 'fc9')
    # # fc10 = fc(fc6, 50, 10, relu = True,name = 'fc10')
    # # dropout7 = dropout(fc7, self.KEEP_PROB)

    # # 8th Layer: FC and return unscaled activations
    # # (for tf.nn.softmax_cross_entropy_with_logits)
    # self.y = fc(fc9, 50, num_out = 1, relu = True, name='fc11')

    #Alex Net Testing End
    #VGG Net Testing
    #  conv1: 24 filters / 5x5 kernel / 2x2 stride
    # conv1 = tf.contrib.layers.conv2d(self.x,64, 3, 2, activation_fn = tf.nn.relu)
    # # conv2: 36 filters / 5x5 kernel / 2x2 stride
    # conv2 = tf.contrib.layers.conv2d(conv1, 64, 3, 2, activation_fn = tf.nn.relu)
    # # conv1: 48 filters / 5x5 kernel / 2x2 stride
    # conv3 = tf.contrib.layers.conv2d(conv2, 128, 3, 2, activation_fn = tf.nn.relu)
    # # conv1: 24 filters / 3x3 kernel / 1x1 stride
    # conv4 = tf.contrib.layers.conv2d(conv3, 128, 3, 1, activation_fn = tf.nn.relu)
    # # conv1: 24 filters / 3x3 kernel / 1x1 stride
    # conv5 = tf.contrib.layers.conv2d(conv4, 256, 3, 1, activation_fn = tf.nn.relu)
    # conv6 = tf.contrib.layers.conv2d(conv5, 256, 3, 1, activation_fn = tf.nn.relu)
    # conv7 = tf.contrib.layers.conv2d(conv6, 256, 3, 1, activation_fn = tf.nn.relu)
    # conv8 = tf.contrib.layers.conv2d(conv7, 512, 3, 1, activation_fn = tf.nn.relu)
    # conv9 = tf.contrib.layers.conv2d(conv8, 512, 3, 1, activation_fn = tf.nn.relu)
    # conv10 = tf.contrib.layers.conv2d(conv9, 512, 3, 1, activation_fn = tf.nn.relu)
    # conv11 = tf.contrib.layers.conv2d(conv10, 512, 3, 1, activation_fn = tf.nn.relu)
    # conv12 = tf.contrib.layers.conv2d(conv11, 512, 3, 1, activation_fn = tf.nn.relu)
    # conv13 = tf.contrib.layers.conv2d(conv12, 512, 3, 1, activation_fn = tf.nn.relu)
    # # To extract features
    # self.conv13 = conv13

    # # fully connected layers
    # flattened = tf.contrib.layers.flatten(conv13)
    # # fc1       = tf.contrib.layers.fully_connected(flattened,  4096, activation_fn = tf.nn.relu)
    # fc2       = tf.contrib.layers.fully_connected(flattened,  1164, activation_fn = tf.nn.relu)
    # fc3       = tf.contrib.layers.fully_connected(fc2,        100,  activation_fn = tf.nn.relu)
    # fc4       = tf.contrib.layers.fully_connected(fc3,        50,   activation_fn = tf.nn.relu)
    # fc5       = tf.contrib.layers.fully_connected(fc4,        10,   activation_fn = tf.nn.relu)
    # self.y    = tf.contrib.layers.fully_connected(fc5,        1,    activation_fn = None)
    
    # VGG Net End

    self.l1        = tf.reduce_mean( tf.abs( self.y_ - self.y ) )
    # self.train_op  = tf.train.AdamOptimizer(config.lr).minimize(self.l1)
    # globa_stp = tf.Variable(4, name='global_step', trainable= False)
    self.train_op  = tf.train.AdamOptimizer().minimize(self.l1)
    #changed for CapsNet testing
    # self.train_op  = tf.train.GradientDescentOptimizer(config.lr).minimize(self.l1)

    # Initialize
    self.sess.run(tf.global_variables_initializer())

  # optimize
  def process(self, sess, x, y_):
    feed = {self.x: x, self.y_: y_}
    return sess.run([self.l1, self.train_op], feed)

  # prediction
  def predict(self, sess, x):
    feed = {self.x: x}
    return sess.run([self.y], feed)

  # validation
  def validate(self, sess, x, y_):
    feed = {self.x: x, self.y_: y_}
    return sess.run([self.l1, self.y], feed)

  # extract features
  def extractFeats(self, sess, x):
    feed = {self.x: x}
    return sess.run([self.conv5], feed)



class VA(object):
  def __init__(self, prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):

    #
    self.prev2out = prev2out
    self.ctx2out  = ctx2out
    self.alpha_c  = alpha_c
    self.selector = selector
    self.dropout  = dropout

    # Parameters
    self.T = config.timelen -1
    self.H = config.dim_hidden
    self.L = config.ctx_shape[0]
    self.D = config.ctx_shape[1]
    self.M = config.dim_hidden
    self.V = 1

    # Place holders
    self.features = tf.placeholder(tf.float32,  [None, self.L, self.D])
    self.y        = tf.placeholder(tf.float32,  [None, 1])

    # Initializer
    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer  = tf.constant_initializer(0.0)

  def _get_initial_lstm(self, features):
    with tf.variable_scope('initial_lstm'):
      features_mean = tf.reduce_mean(features, 1)

      w_h = tf.get_variable('w_h', [self.D, self.H],  initializer=self.weight_initializer)
      b_h = tf.get_variable('b_h', [self.H],          initializer=self.const_initializer)
      h   = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

      w_c = tf.get_variable('w_c', [self.D, self.H],  initializer=self.weight_initializer)
      b_c = tf.get_variable('b_c', [self.H],          initializer=self.const_initializer)
      c   = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
      return c, h

  def _project_features(self, features):
    with tf.variable_scope('project_features'):
      w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
      features_flat = tf.reshape(features, [-1, self.D])
      features_proj = tf.matmul(features_flat, w)
      features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
      return features_proj

  def _attention_layer(self, features, features_proj, h, reuse=False):
    with tf.variable_scope('attention_layer', reuse=reuse):
      w     = tf.get_variable('w',     [self.H, self.D], initializer=self.weight_initializer)
      b     = tf.get_variable('b',     [self.D],         initializer=self.const_initializer)
      w_att = tf.get_variable('w_att', [self.D, 1],      initializer=self.weight_initializer)

      h_att   = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b) 
      out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])
      alpha   = tf.nn.softmax(out_att)
      context = tf.reshape(features * tf.expand_dims(alpha, 2), [-1, self.L*self.D])
      return context, alpha

  def _selector(self, context, h, reuse=False):
    with tf.variable_scope('selector', reuse=reuse):
      w       = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
      b       = tf.get_variable('b', [1], initializer=self.const_initializer)
      beta    = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta') 
      context = tf.multiply(beta, context, name='selected_context')
      return context, beta

  def _decode_lstm(self, h, context, dropout=False, reuse=False):
    with tf.variable_scope('logits', reuse=reuse):
      w_h   = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
      b_h   = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
      w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
      b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

      if dropout:
        h = tf.nn.dropout(h, 0.5)
      h_logits = tf.matmul(h, w_h) + b_h

      if self.ctx2out:
        w_ctx2out = tf.get_variable('w_ctx2out', [self.L*self.D, self.M], initializer=self.weight_initializer)
        h_logits += tf.matmul(context, w_ctx2out)

      if dropout:
        h_logits = tf.nn.dropout(h_logits, 0.5)
      out_logits = tf.matmul(h_logits, w_out) + b_out
      return out_logits

  def _batch_norm(self, x, mode='train', name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

  def build_model(self):
    features    = self.features
    y           = self.y
    batch_size  = tf.shape(features)[0]

    # batch normalize feature vectors
    features      = self._batch_norm(features, mode='train', name='conv_features')

    # features to initialize
    gather_indices_init  = tf.range(config.epoch) * config.timelen
    features_init = tf.gather( features, gather_indices_init )

    c, h          = self._get_initial_lstm(features=features_init)
    features_proj = self._project_features(features=features)

    loss        = 0.0
    alpha_list  = []
    lstm_cell   = tf.nn.rnn_cell.LSTMCell(name = 'basic_lstm_cell',num_units=self.H)

    for t in range(self.T+1):
      gather_indices  = tf.range(config.epoch) * config.timelen + t 
      features_curr   = tf.gather( features, gather_indices ) 
      features_proj_curr = tf.gather( features_proj, gather_indices ) 
      y_curr          = tf.gather( y, gather_indices ) 

      context, alpha = self._attention_layer(features_curr, features_proj_curr, h, reuse=(t!=0)) 
      alpha_list.append(alpha) 

      if self.selector:
        context, beta = self._selector(context, h, reuse=(t!=0))

      with tf.variable_scope('lstm', reuse=(t!=0)):
        _, (c, h) = lstm_cell(inputs=tf.concat([context],1), state=[c, h])

      logits = self._decode_lstm(h, context, dropout=self.dropout, reuse=(t!=0))

      loss += tf.reduce_sum( tf.abs( tf.subtract( logits, y_curr ) ) ) 

    if self.alpha_c > 0:
      alphas      = tf.transpose(tf.pack(alpha_list), (1, 0, 2))
      alphas_all  = tf.reduce_sum(alphas, 1)
      alpha_reg   = self.alpha_c * tf.reduce_sum((config.timelen/(config.ctx_shape[0]*config.ctx_shape[1]) - alphas_all) ** 2)  # timestep/#activations
      loss       += alpha_reg

    return loss / tf.to_float(config.epoch)

  def build_sampler(self, max_len=20):
    features = self.features
    features = self._batch_norm(features, mode='test', name='conv_features')

    gather_indices_init  = tf.range(config.epoch) * config.timelen
    features_init = tf.gather( features, gather_indices_init )

    c, h            = self._get_initial_lstm(features=features_init)
    features_proj   = self._project_features(features=features)

    y_list      = []
    alpha_list  = []
    beta_list   = []
    lstm_cell   = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

    for t in range(self.T+1):
      gather_indices  = tf.range(config.epoch) * config.timelen + t
      features_curr   = tf.gather( features, gather_indices )
      features_proj_curr = tf.gather( features_proj, gather_indices )

      context, alpha = self._attention_layer(features_curr, features_proj_curr, h, reuse=(t!=0))
      alpha_list.append(alpha)


      if self.selector:
        context, beta = self._selector(context, h, reuse=(t!=0))
        beta_list.append(beta)

      with tf.variable_scope('lstm', reuse=(t!=0)):
        _, (c, h) = lstm_cell(inputs=tf.concat(1, [context]), state=[c, h])

      logits = self._decode_lstm(h, context, dropout=False, reuse=(t!=0))
      y_list.append(logits)

    alphas  = tf.transpose(tf.pack(alpha_list), (1, 0, 2))
    betas   = tf.transpose(beta_list)
    ys      = tf.squeeze(y_list)
    return alphas, betas, ys


