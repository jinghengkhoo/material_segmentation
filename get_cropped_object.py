
import json 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

f = open('room.json')
orig_img = cv2.imread('room.jpg')
mask = np.zeros_like(orig_img) 

data = json.load(f)

categories = data['categories']
annots = data['annotations']

def make_contour(segments_pts):
    points_arr = []
    i = 0
    
    while i < len(segments_pts):
        arr = []
        x_point = segments_pts[i]
        arr.append(x_point)
        i+=1
        y_point = segments_pts[i]
        i+=1
        arr.append(y_point)
        points_arr.append(arr)
    
    points = np.array(points_arr)
    
    return points


for i in range(len(annots)):
    segments_Points=annots[i]['segmentation']
    category_id = annots[i]['category_id']
    crop_region = categories[category_id]['name']
    mask = np.zeros_like(orig_img) 

    for segments in range(len(segments_Points)):
        
        pts = make_contour(segments_Points[segments])
                
        cv2.fillPoly(mask, np.int32([pts]), color = (255,255,255))
        result = cv2.bitwise_and(orig_img, mask)
	change_mask = result[:,:,2] ==0
        result[change_mask]=[255, 255, 255]

        cv2.imwrite('results/'+ str(crop_region) + '.jpg', result)

        

        
  




