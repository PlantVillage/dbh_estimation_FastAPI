import numpy as np
import cv2
import traceback
from itertools import groupby
from matplotlib import gridspec
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imutils
from imutils import perspective

# define pixel constants
trunk = [0,85,0]
#tag  = [100,150,255]
tag  = [255,150,100]
white = [255,255,255]
black = [0,0,0]

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

def generateVisualization(seg_image, x,y, avg_tree_pixel_width, w , file, indexes, box, measured_dbh, predicted_dbh):
    output_path = f'data/outputs/overlay_{file}.png'
    fig = plt.figure(figsize=(60, 20))
    grid_spec = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1])


    # show mask 
    mask_location = 'data/outputs/resized_original_img_1.png' #'data/outputs/temp.png'
    mask = Image.open(mask_location)
    plt.subplot(grid_spec[0])
    plt.title(f'Original Image (Measured dbh = {measured_dbh})', fontdict = {'fontsize' : 30})
    plt.imshow(mask)
    plt.axis('off')

    # show mask 
    mask_location = 'data/outputs/seg_image_original_1.png' #'data/outputs/temp.png'
    mask = Image.open(mask_location)
    plt.subplot(grid_spec[1])
    plt.title('Original Segmetation Mask', fontdict = {'fontsize' : 30})
    plt.imshow(mask)
    plt.axis('off')

    # show  resized image
    filename =  'data/outputs/resized_original_img.png' #'data/outputs/resized_img.png'
    img = Image.open(filename)
    plt.subplot(grid_spec[2])
    plt.title('Resized Image', fontdict = {'fontsize' : 30})
    plt.imshow(img)
    plt.axis('off')

    # show mask overlay image
    alpha = 0.6
    plt.subplot(grid_spec[3])
    plt.title('Segmentation Image Overlay', fontdict = {'fontsize' : 30})
    plt.imshow(img)
    plt.imshow(seg_image, alpha=alpha)
    plt.axis('off')

    # show mask ovelay on image with tag and trunk pixel widht estimations
    plt.subplot(grid_spec[4])
    plt.title(f'Pixel Width Overlay (Predicted dbh = {predicted_dbh})', fontdict = {'fontsize' : 30})
    DrawImage = ImageDraw.Draw(img)
    # draw tag width estimation
    DrawImage.line([(int(x-(w/2)), int(y+(w/2))),(int(x+(w/2)), int(y+(w/2)))], fill="red", width=5)
    # draw tree trunk width estimation

    # draw tag box
    (tl, tr, br, bl) = box
    DrawImage.line([tuple(tl), tuple(tr)], fill='blue', width=3)
    DrawImage.line([tuple(tr), tuple(br)], fill='blue', width=3)
    DrawImage.line([tuple(br), tuple(bl)], fill='blue', width=3)
    DrawImage.line([tuple(bl), tuple(tl)], fill='blue', width=3)

    tree_x1 = indexes[0]
    wt = avg_tree_pixel_width
    DrawImage.line([(tree_x1 , int(y-(w/2))),( tree_x1 + wt , int(y-(w/2)))], fill="red", width=5)
    plt.imshow(img)
    plt.imshow(seg_image, alpha=alpha)

    # save overlay images
    plt.savefig(output_path)

def getTreePixelLenght(y, seg_image, x, buffer):
    ''' 
    Takes cordinates for the top of the tag (y) and the middle of the tag (x on the x-axis) and return the pixel tree width at the row y-buffer. It also return the coordinates for the start and end of the tree pixels on the x-axis
    '''
    x = int(x)
    def getLength(indexes):
        return indexes[1] - indexes[0]

    try:
        row = seg_image[y-buffer]
        adjacent_trunks = find_continuous_indexes(row, trunk)
        #print(adjacent_trunks)
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

def getRangTreePixelLengths(y1, y2, seg_image, x):

    ''' 
    Gets the average tree pixel width over a range of y values from y1 to y2, which intersects with the x 
    '''
    try:
        range_length = []
        indexes = []
        for i in np.arange(y1, y2):
            temp_lenth, temp_index = getTreePixelLenght(i, seg_image, x, 0)
            range_length.append(temp_lenth)
            indexes.append(temp_index)

        if len(range_length) != 0:
            tree_width =  np.mean(range_length)
            x_pixel = np.mean([int(x) for (x,_) in indexes])
            y_pixel = np.mean([int(y) for (_,y) in indexes])
            return [tree_width, (x_pixel, y_pixel)]

    except:
        print(traceback.format_exc())
        return None


def getTreePixelWidth(seg_image, file, measured_dbh, tag_width):
  buffer = 10 # pixel buffer. It avoids the instance of having the tag pixels in the list  
  try:
    
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
        #avg_tree_pixel_width = getTreePixelLenght(y3, seg_image, x, buffer)
        avg_tree_pixel_width = getRangTreePixelLengths(y3,y4, seg_image, x)
        #print(avg_tree_pixel_width)
        #getRangTreePixelLengths
    elif y3 == y4: # average pixels above tag
        #avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y1, y2)])
        #avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x, buffer)
        avg_tree_pixel_width = getRangTreePixelLengths(y1,y2, seg_image, x)
        #print(avg_tree_pixel_width)
    else:
        #top_avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x, buffer) 
        top_avg_tree_pixel_width = getRangTreePixelLengths(y1,y2, seg_image, x)
        #bottom_avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y3, y4)])
        avg_tree_pixel_width = top_avg_tree_pixel_width
        #print(avg_tree_pixel_width)

    if avg_tree_pixel_width == None:
        avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x, buffer) 

    predicted_dbh = round((avg_tree_pixel_width[0]/w) * tag_width,2)

    generateVisualization(seg_image, x,y, avg_tree_pixel_width[0] ,w, file, avg_tree_pixel_width[1], box, measured_dbh, predicted_dbh)
  
    return avg_tree_pixel_width[0]/w

  except:
    print(traceback.format_exc())
    return None 

def getZoomCordinates(seg_image, buffer_pixels):
    buffer = 10
    left = top = 0 ; right = len(seg_image[0]) ; bottom = len(seg_image) # initialize cordinates with sensible values

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
        avg_tree_pixel_width = getTreePixelLenght(y3, seg_image, x, buffer)
    elif y3 == y4: # average pixels above tag
        #avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y1, y2)])
        avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x, buffer)
    else:
        top_avg_tree_pixel_width = getTreePixelLenght(y2, seg_image, x, buffer) 
        #bottom_avg_tree_pixel_width = np.mean([getTreePixelLenght(i, seg_image, x) for i in range(y3, y4)])
        avg_tree_pixel_width = top_avg_tree_pixel_width 
    
    box = perspective.order_points(box)
    (tl, _, br, _) = box

    tree_width = avg_tree_pixel_width[1][1] - avg_tree_pixel_width[1][0]

    # best for bigger trees
    left = max(avg_tree_pixel_width[1][0] - (tree_width * 0.5), left)  # ensure that it is not negative
    right = min(avg_tree_pixel_width[1][1] + (tree_width * 0.5), right)

    # if the tree is small
    if tree_width/w < 1.5:
        # best for bigger trees
        print("small tree")
        left = avg_tree_pixel_width[1][0] - (h * 1.5)  # ensure that it is not negative
        right = avg_tree_pixel_width[1][1] + (h * 1.5)


    tag_top = int(y - (h/2)) ; tag_bottom =  int(y + (h/2))

    top = max(tag_top - (h * 2), top)  # ensure that it is not negative
    bottom = min( tag_bottom + (h * 2), bottom)
    
    #print(left, top, right, bottom)
    
    #left = avg_tree_pixel_width[1][0] - buffer_pixels
    #top = int(tl[0]) - (buffer_pixels*2)
    #bottom = int(br[0]) + (buffer_pixels*2)
    #right = avg_tree_pixel_width[1][1] + buffer_pixels

    # ensure that image is portrait
    while right -left > bottom - top:
        bottom = bottom + (buffer_pixels *0.5)

    #print(left,top,right,bottom )

    return left,top,right,bottom 