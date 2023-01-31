import numpy as np
import cv2
import traceback
from itertools import groupby



# define pixel constants
trunk = [0,85,0]
#tag  = [100,150,255]
tag  = [255,150,100]
white = [255,255,255]
black = [0,0,0]

def getTagAndTrunkPixelsWidth(seg_map): 

    ''' 
    Takes model pixels predictions and return estimation of tag and tree trunk width
    '''

    trunk = [0,85,0]
    tag  = [255,150,100]
    #white = [255,255,255]
    #black = [0,0,0]

    '''
    def getTagAndTrunkWidth(row):
        return {
            "tag_width": row.tolist().count(3),
            "trunk_width": row.tolist().count(2)
        }
    '''

    def getTagAndTrunkWidth(row):
      return {
          "tag_width": row.tolist().count(tag),
          "trunk_width": row.tolist().count(trunk)
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
  try:
    tag_im = getTagMask(seg_image)
    tag_contours = getContour(tag_im)
    #print(tag_contours)
    return getPixelsPerMetricHelper(tag_contours[getTagContour(tag_contours)], tag_width)
  except:
    print(traceback.format_exc())
    return None


def getTreePixelWidth(seg_image):
  def groups(l):
    return [sum(g) for i, g in groupby(l) if i == 1]

  def getTagAdjacentTreePixelLength(pixel_length):
    return np.max(pixel_length)
  
  # loop over the rows and determine the number of pixels that are tree trunk
  def getTreePixelLenght(temp, im):
      tree_pixels = [1 if pixel == trunk else 0 for pixel in im[temp].tolist()]
      tree_pixels_lengths = groups(tree_pixels)
      tag_adjacent_tree_pixel = getTagAdjacentTreePixelLength(tree_pixels_lengths)
      return tag_adjacent_tree_pixel
  
  y1 = y3 = 0; y2 = y4 = len(seg_image) # use the average tree pixel length all over the image

  try: # try to get the average pixel length just above  and below the tag
    # can make this global and utilize aleady calculated one in the pixel estimation funciton
    tag_im = getTagMask(seg_image)
    tag_contours = getContour(tag_im)

    c = tag_contours[getTagContour(tag_contours)]

    box = cv2.minAreaRect(c)
    (_,_), (_,h), _ = box
    #box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    #box = np.array(box, dtype="int")

    # determine y1, y2 - scenario 1 (best case tag is right in the middle of bottom of the image)
    y2 = np.min([int(y) for (_,y) in box.tolist()]) # find y cordintate for the tag position
    y1 = y2 - int(h)

    if y1 < 0: # if tag happens to be at the top of the image
        y1=y2

    y3 = y2 = np.max([int(y) for (_,y) in box.tolist()])
    y4 = y3  + int(h)

    if y4 > len(seg_image):
        y4 = y3
  except:
    pass # if it fails we just continue 

  try: # try to averae pixels above and below tag

    top_avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image) for i in range(y1, y2)])
    bottom_avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image) for i in range(y3, y4)])

    avg_tree_pixel_width = (top_avg_tree_pixel_width + bottom_avg_tree_pixel_width)/2
  except:
    if y1 ==y2: # average pixels below tag
        avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image) for i in range(y3, y4)])
    elif y3 == y4: # average pixels above tag
        avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image) for i in range(y1, y2)])
    else: # average pixels all over the image
        avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image) for i in range(y1, y2)])

  return avg_tree_pixel_width