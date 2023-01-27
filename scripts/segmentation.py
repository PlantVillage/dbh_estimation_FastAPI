# Commented out IPython magic to ensure Python compatibility.
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import PIL

# %tensorflow_version 1.x
import tensorflow.compat.v1 as tf
import math

import traceback

domain = 'tree_trunk'
## model path


path_to_model = '/Users/edwardamoah/Documents/GitHub/tree_semantic_segmentation/models/tree_trunk/tree_trunk_frozen_graph_1.4.pb'

#image_path = "/Users/edwardamoah/Documents/GitHub/tree_semantic_segmentation/data/images_crop/1661327874369.jpg"


#### load DeepLab Model #####
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513

  def __init__(self, file_handle):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_handle, 'rb') as fid:
      serialized_graph = fid.read()
      graph_def.ParseFromString(serialized_graph)

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.
    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map




def create_tree_trunk_label_colormap():
  """Creates a label colormap for the locusts dataset.
  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((4, 3), dtype=int)
  colormap[0] = [0,0,0]
  colormap[1] = [255,255,255]
  colormap[2] = [0,85,0]
  colormap[3] = [255,150,100]
  return colormap

def label_to_color_image(label, domain):
  """
  Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
    domain: A string specifying which label map to use
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  elif domain == 'tree_trunk':
    colormap = create_tree_trunk_label_colormap()
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def get_label_names(domain):
  if domain == 'tree_trunk': #dumby labels
    LABEL_NAMES = np.asarray([ "Unlabeled", "Background", "Tree trunk", "Tag"
    ])
  else:
    LABEL_NAMES = 'error'

  return LABEL_NAMES

label_names = get_label_names(domain)
FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, domain)



def getTagAndTrunkPixelsWidth(seg_map): 

    ''' 
    Takes model pixels predictions and return estimation of tag and tree trunk width
    '''

    def getTagAndTrunkWidth(row):
        return {
            "tag_width": row.tolist().count(3),
            "trunk_width": row.tolist().count(2)
        }

    def getTagRows(seg_map):
        tag_rows = []
        for i in range(len(seg_map)):
            row = seg_map[i]
            #print(row)
            if getTagAndTrunkWidth(row)["tag_width"] > 0:
                tag_rows.append(i)

        def continuous_list(list):
            result_list = []
            for i in range(len(list)):
                if i+1 < len(list) and list[i]+1 == list[i+1]:
                    result_list.append(list[i])
            return result_list

        tag_rows = continuous_list(tag_rows) # make sure tag pixels are continuous
        
        return [min(tag_rows),max(tag_rows)] 

    y1, y2 = getTagRows(seg_map) # get tag rows

    # get tag width
    tag_widths = []
    for i in range(y1,y2):
        tag_widths.append(getTagAndTrunkWidth(seg_map[i])["tag_width"])

    
    # work on tree pixels
    y0 = max(y1-(y2-y1),0)
    y3 = min(len(seg_map), y2+(y2-y1))

    trunk_widths = []
    for i in range(y0,y1):
        trunk_widths.append(getTagAndTrunkWidth(seg_map[i])["trunk_width"])
    for i in range(y2,y3):
        trunk_widths.append(getTagAndTrunkWidth(seg_map[i])["trunk_width"])

 
    return {
        "tag": np.mean(tag_widths),
        "trunk": np.mean(trunk_widths)
    }


from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import imutils
import cv2


# define pixels
trunk = [0,85,0]
#tag  = [100,150,255]
tag  = [255,150,100]
white = [255,255,255]
black = [0,0,0]

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def getTreeMask(im):
    tree_im = im.copy()
    # Make background black
    tree_im[np.all(tree_im == white, axis=-1)] = black

    # make tag black
    tree_im[np.all(tree_im == tag, axis=-1)] = black

    # make trunk white
    tree_im[np.all(tree_im == trunk, axis=-1)] = white

    # return tree_mask
    return tree_im

# get tree mask
#tree_im = getTreeMask(im)
#plt.imshow(tree_im)

def getTagMask(im):
    tag_im = im.copy()
    # Make background black
    tag_im[np.all(tag_im == white, axis=-1)] = black

    # make trunk black
    tag_im[np.all(tag_im == trunk, axis=-1)] = black

    # make tag white
    tag_im[np.all(tag_im == tag, axis=-1)] = white

    return tag_im


def getContour(mask):
    # get the edges
    edged = cv2.Canny(mask, 50, 100)

    # get the contours 
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours


def getTagContour(tag_contours):
    ''' 
    Find and return index of the tag contour with lagest area
    '''
    tag_area = [cv2.contourArea(c) for c in tag_contours]
    return tag_area.index(np.max(tag_area))

def getPixelsPerMetricHelper(tag_contour, metric):

    ''' 
    Takes tag contour and return pixelsPerMetric base on X pixel, Y pixels widths
    '''

    # get miminumum box
    box = cv2.minAreaRect(tag_contour)
    (_,_), (w,_), _ = box

    # return 
    return w/metric


def getPixelPerMetric(seg_image, tag_width):
    tag_im = getTagMask(seg_image)
    tag_contours = getContour(tag_im)
    #print(tag_contours)
    return getPixelsPerMetricHelper(tag_contours[getTagContour(tag_contours)], tag_width)


MODEL = DeepLabModel(path_to_model)
def getTreeDBH(filename, tag_width):
    try:
        # load image
        im = Image.open(filename)

        # pre-process image 
          # determine if image needs to be rotated and rotate it
          # determine if image needs to be zoomed in and do so
          # 

        # run model 
        _, seg_map = MODEL.run(im)
        seg_image = label_to_color_image(seg_map, domain).astype(np.uint8)

        # move to background later -- saved mask
        new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
        new_seg_iamge.save('data/outputs/temp.png')

        # get pixels per cm value for the image
        pixelsPerMetric = getPixelPerMetric(seg_image, tag_width)

        # get pixel width for tree
        pixels_width = getTagAndTrunkPixelsWidth(seg_map)["trunk"]
        
        dbh = pixels_width/pixelsPerMetric

        return dbh

    except Exception:
        return None