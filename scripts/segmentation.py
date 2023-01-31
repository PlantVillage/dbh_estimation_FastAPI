# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from PIL import Image
import PIL
import traceback
from scripts import pixel_analyzer as pa
from scripts import deeplab_model, runTilles


domain = 'tree_trunk'

MODEL = deeplab_model.MODEL # load from deeplab_script

'''
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
'''

def getTreeDBH(filename, tag_width):
    try:
        # load image
        im = Image.open(filename)

        # check image orientation and flip if needed
        width, height = im.size
        if width > height:
          im = im.rotate(270, PIL.Image.NEAREST, expand = 1)

        # run model 
        _ , seg_map = MODEL.run(im)
        seg_image = deeplab_model.label_to_color_image(seg_map, domain).astype(np.uint8)

        # move to background later -- saved mask
        new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
        new_seg_iamge.save('data/outputs/temp.png')

        # get pixels per cm value for the image
        pixelsPerMetric = pa.getPixelPerMetric(seg_image, tag_width)
        #print(pixelsPerMetric)

        # if no tag is detected run the image as tiles
        if pixelsPerMetric == None:
          seg_image = runTilles.runTilles(filename)
          pixelsPerMetric = pa.getPixelPerMetric(seg_image, tag_width)

          # move to background later -- saved mask
          new_seg_iamge = Image.fromarray(np.uint8(seg_image)).convert('RGB')
          new_seg_iamge.save('data/outputs/temp.png')

        # check if the tag detected is just noise
        if pixelsPerMetric != None and pixelsPerMetric < 1:
          return 'No tag was detected'

        # get pixel width of tree around the tag
        pixels_width = pa.getTreePixelWidth(seg_image) 
        
        # ESTIMATE DBH !!!
        dbh = pixels_width/pixelsPerMetric

        return dbh

    except Exception:
        print(traceback.format_exc())
        return None