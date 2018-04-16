import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util
#from utils import draw_bounding_boxes_on_image


def show_webcam(mirror=False):
  cap = cv2.VideoCapture(1)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('./media/test.avi',fourcc, 20.0, (640,480))
  #out = cv2.VideoWriter('./media/test.avi', -1,1,(640,480))
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
      #frame = cv2.flip(frame,0)
      out.write(frame)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break
#for j in sorted(os.listdir('/home/adithya/Downloads/google-images-download/google_images_download/downloads/aamir khan taare zameen par')):
#print(j)
#img = cv2.imread('/home/adithya/Downloads/google-images-download/google_images_download/downloads/aamir khan taare zameen par/'+ j)
#res = cv2.resize(img,(640,480),interpolation = cv2.INTER_CUBIC)
#cv2.imshow('image',res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#out.write(res)
  cap.release()
  out.release()
  cv2.destroyAllWindows()


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (64,64)
import matplotlib.image as mpimg

show_webcam(mirror = True)
cap = cv2.VideoCapture("./media/test.avi")
#cap = show_webcam(mirror = True)
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490;
    j = 0 
    while(frame_num):
      frame_num -= 1
      ret, image = cap.read()
      v = image
      if(ret ==0):
          break

      if out is None:
          [h, w] = image.shape[:2]
          out = cv2.VideoWriter("./media/test_out.avi", 0, 25.0, (w, h))


      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      #print(boxes)
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)
      # Visualization of the results of a detection.
      image_extracted = vis_util.visualize_boxes_and_labels_on_image_array(
#          image_np,
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      out.write(image)
      #vis_util.draw_bounding_boxes_on_image(v,np.squeeze(boxes))
      #plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image)
      if type(image_extracted) is np.ndarray:
        plt.imsave(('./media/test_op_11' + '_output%s.png')%(str(j)),image_extracted)

      for w in range(len(image_extracted)):
        plt.imsave(('./media/test_op_11' + '_output%s.png')%(str(w) + '*' +str(j)),image_extracted[w])

      j = j + 1
      # plt.imsave("_output.png", image_np)


    cap.release()
    out.release()
