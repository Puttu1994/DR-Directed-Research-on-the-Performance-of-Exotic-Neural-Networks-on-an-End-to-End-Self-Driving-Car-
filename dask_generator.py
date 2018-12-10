
import numpy as np
import h5py
import time
import logging
import traceback

from    utils.config        import *
import  scipy.ndimage       as ndi

logger = logging.getLogger(__name__)

# this is non-causal filter! you should use causal filter!!
def gaussian_smoothing(x, sigma):
  return ndi.gaussian_filter1d(x, sigma=sigma, output=np.float64, mode='nearest')

# given a series and alpha, return series of smoothed points
def double_exponential_smoothing(series, alpha, beta):
  result = [series[0]]
  for n in range(1, len(series)+1):
    if n == 1:
      level, trend = series[0], series[1] - series[0]
    if n >= len(series): # we are forecasting
      value = result[-1]
    else:
      value = series[n]
    last_level, level = level, alpha*value + (1-alpha)*(level+trend)
    trend = beta*(level-last_level) + (1-beta)*trend
    result.append(level+trend)
  return result

# given a series and alpha, return series of smoothed points
def exponential_smoothing(series, alpha):
  result = [series[0]] # first value is same as series
  for n in range(1, len(series)):
    result.append(alpha * series[n] + (1 - alpha) * result[n-1])
  return np.array(result)

def concatenate(camera_names, time_len):

  if config.UseFeat:
    logs_names = [x.replace('features', 'log') for x in camera_names]
  else:
    logs_names = [x.replace('camera', 'log') for x in camera_names]

  angle = []  # steering angle of the car
  speed = []  # steering angle of the car
  hdf5_camera = []  # the camera hdf5 files need to continue open
  c5x = []
  filters = []
  lastidx = 0

  for cword, tword in zip(camera_names, logs_names):
    try:
      with h5py.File(tword, "r") as t5:
        c5 = h5py.File(cword, "r")
        hdf5_camera.append(c5)
        x = c5["X"]
        c5x.append((lastidx, lastidx+x.shape[0], x))

        # Measured in 60Hz
        speed_value     = t5["speed"][:]
        steering_angle  = t5["steering_angle"][:]

        # Smoothing function
        if config.use_smoothing == "Exp": # Single Exponential Smoothing
          print("Exp Smoothing...", config.use_smoothing)
          speed_value    = exponential_smoothing(speed_value,     config.alpha)
          steering_angle = exponential_smoothing(steering_angle,  config.alpha)

        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
        angle.append(steering_angle[idxs])
        speed.append(speed_value[idxs])

        # Choose good imgages?
        goods = (np.abs(angle[-1]) <= 200) & (np.abs(speed[-1]) > 1)

        filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
        lastidx += goods.shape[0]
        # check for mismatched length bug
        print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
        if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
          raise Exception("bad shape")

    except IOError:
      import traceback
      traceback.print_exc()
      print ("failed to open", tword)

  angle = np.concatenate(angle, axis=0)
  speed = np.concatenate(speed, axis=0)
  filters = np.concatenate(filters, axis=0).ravel()
  print ("training on %d/%d examples"%(filters.shape[0],angle.shape[0]))
  return c5x, angle, speed, filters, hdf5_camera


first = True


def datagen(filter_files, time_len=1, batch_size=256, ignore_goods=False):
  """
  Parameters:
  -----------
  leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
  """
  global first
  assert time_len > 0
  filter_names = sorted(filter_files)

  logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))

  c5x, angle, speed, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
  filters_set = set(filters)

  logger.info("camera files {}".format(len(c5x)))

  if config.UseFeat:
    X_batch = np.zeros((batch_size, time_len, 64, 10, 20), dtype='uint8')
  else:
    X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')

  angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

  while True:
    try:
      t = time.time()

      count = 0
      start = time.time()
      while count < batch_size:
        if not ignore_goods:
          i = np.random.choice(filters)
          # check the time history for goods
          good = True
          for j in (i-time_len+1, i+1):
            if j not in filters_set:
              good = False
          if not good:
            continue

        else:
          i = np.random.randint(time_len+1, len(angle), 1)

        # GET X_BATCH
        # low quality loop
        for es, ee, x in c5x:
          if i >= es and i < ee:
            X_batch[count] = x[i-es-time_len+1:i-es+1]
            break

        angle_batch[count] = np.copy(angle[i-time_len+1:i+1])[:, None]
        speed_batch[count] = np.copy(speed[i-time_len+1:i+1])[:, None]

        count += 1

      # sanity check
      if config.UseFeat:
        assert X_batch.shape == (batch_size, time_len, 64, 10, 20)
      elif config.CausalityTest:
        assert X_batch.shape == (batch_size, time_len, 64, 10, 20)
      else:
        assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

      logging.debug("load image : {}s".format(time.time()-t))
      print("%5.2f ms" % ((time.time()-start)*1000.0))
      print(first)

      if first:
        print ("X",(X_batch.shape))
        print ("angle",(angle_batch.shape))
        print ("speed",(speed_batch.shape))
        first = False
        print (first)
      
      yield(X_batch, angle_batch, speed_batch)
    except KeyboardInterrupt:
      raise
    except:
      traceback.print_exc()
      pass
