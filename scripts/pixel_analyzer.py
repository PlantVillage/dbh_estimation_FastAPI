import numpy as np
import cv2
import traceback
from itertools import groupby
from matplotlib import gridspec
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imutils



# define pixel constants
trunk = [0,85,0]
#tag  = [100,150,255]
tag  = [255,150,100]
white = [255,255,255]
black = [0,0,0]


'''
def getTagAndTrunkPixelsWidth(seg_map): 


    trunk = [0,85,0]
    tag  = [255,150,100]
    #white = [255,255,255]
    #black = [0,0,0]

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

'''




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


def find_continuous_indexes(lst, value):
    result = []
    start_index = None
    end_index = None
    for i in range(len(lst)):
        if lst[i].tolist() == value:
            if start_index is None:
                start_index = i
            end_index = i
        else:
            if start_index is not None:
                result.append((start_index, end_index))
                start_index = None
                end_index = None
    if start_index is not None:
        result.append((start_index, end_index))
    return result

def generateVisualization(seg_image, x,y, avg_tree_pixel_width, w , file, indexes):
            output_path = f'data/outputs/overlay_{file}.png'
            fig = plt.figure(figsize=(50, 20))
            grid_spec = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

            # show  resized image
            filename = 'data/outputs/resized_img.png'
            img = Image.open(filename)
            plt.subplot(grid_spec[0])
            plt.title('Resized Image', fontdict = {'fontsize' : 30})
            plt.imshow(img)
            plt.axis('off')

            # show mask 
            mask_location = 'data/outputs/temp.png'
            mask = Image.open(mask_location)
            plt.subplot(grid_spec[1])
            plt.title('Segmentation Mask', fontdict = {'fontsize' : 30})
            plt.imshow(mask)
            plt.axis('off')

            # show mask overlay image
            alpha = 0.6
            plt.subplot(grid_spec[2])
            plt.title('Segmentation Image Overlay', fontdict = {'fontsize' : 30})
            plt.imshow(img)
            plt.imshow(seg_image, alpha=alpha)
            plt.axis('off')

            # show mask ovelay on image with tag and trunk pixel widht estimations
            plt.subplot(grid_spec[3])
            plt.title('Pixel Width Overlay', fontdict = {'fontsize' : 30})
            DrawImage = ImageDraw.Draw(img)
            # draw tag width estimation
            DrawImage.line([(int(x-(w/2)), int(y+(w/2))),(int(x+(w/2)), int(y+(w/2)))], fill="red", width=5)
            # draw tree trunk width estimation

            tree_x1 = indexes[0]
            wt = avg_tree_pixel_width
            DrawImage.line([(tree_x1 , int(y-(w/2))),( tree_x1 + wt , int(y-(w/2)))], fill="red", width=5)
            plt.imshow(img)
            plt.imshow(seg_image, alpha=alpha)

            # save overlay images
            plt.savefig(output_path)


def getTreePixelWidth(seg_image, file):
  try:
    # loop over the rows and determine the number of pixels that are tree trunk
    def getTreePixelLenght(y, seg_image, x):
        x = int(x)
        def getLength(indexes):
            return indexes[1] - indexes[0]

        try:
            row = seg_image[y]
            adjacent_trunks = find_continuous_indexes(row, trunk)
            for indexes in adjacent_trunks:
                if x in np.arange(indexes[0],indexes[1]):
                    return [getLength(indexes), indexes]


            #return tag_adjacent_tree_pixel
            max_length = np.max([getLength(indexes) for indexes in adjacent_trunks])
            indexes = adjacent_trunks.index(max_length)
            return [max_length, indexes]
        except:
            print(traceback.format_exc())
            return None
    
    y1 = y3 = 0; y2 = y4 = y = int(len(seg_image)/3) # use the average tree pixel length all over the image


    # can make this global and utilize aleady calculated one in the pixel estimation funciton
    tag_im = getTagMask(seg_image)
    tag_contours = getContour(tag_im)

    c = tag_contours[getTagContour(tag_contours)]

    box = cv2.minAreaRect(c)
    (x,y), (w,h), _ = box

    # determine y1, y2 - scenario 1 (best case tag is right in the middle of bottom of the image)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    y2 = np.min([int(y) for (_,y) in box.tolist()]) # find y cordintate for the tag position
    y1 = y2 - int(h)

    if y1 < 0: # if tag happens to be at the top of the image
        y1=y2

    y3 = np.max([int(y) for (_,y) in box.tolist()])
    y4 = y3  + int(h)

    if y4 > len(seg_image):
        y4 = y3

    # calculate pixel length for the tree
    if y1 ==y2: # average pixels below tag
        #avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y3, y4)])
        avg_tree_pixel_width = getTreePixelLenght(y3, seg_image, x)
    elif y3 == y4: # average pixels above tag
        #avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y1, y2)])
        avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x)
    else:
        top_avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x) 
        #bottom_avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y3, y4)])
        avg_tree_pixel_width = top_avg_tree_pixel_width 

    generateVisualization(seg_image, x,y, avg_tree_pixel_width[0] ,w, file, avg_tree_pixel_width[1])
  
    return avg_tree_pixel_width[0]/w
    
  except:
    print(traceback.format_exc())
    return None 