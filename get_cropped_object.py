
import json 
import numpy as np
import cv2



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

def get_crops(json_mask, orig_img):
    mask = np.zeros_like(orig_img) 

    categories = json_mask['categories']
    annots = json_mask['annotations']

    height, width, _ = orig_img.shape

    crops = {}
    bbox = {}

    for i in range(len(annots)):
        segments_Points=annots[i]['segmentation']
        category_id = annots[i]['category_id']
        crop_region = categories[category_id]['name']
        mask = np.zeros_like(orig_img) 

        for segments in range(len(segments_Points)):
            
            pts = make_contour(segments_Points[segments])
                    
            cv2.fillPoly(mask, np.int32([pts]), color = (255,255,255))
            result = cv2.bitwise_and(orig_img, mask)
            change_mask = result[:,:,2] == 0
            result[change_mask] = [255,255,255]
            crops[crop_region] = result

    for name, crop in crops.items():
        out = np.zeros((height, width), np.uint16)
        for i in range(1, height): 
            for j in range(1, width):
                if (np.any(crop[i][j] < 255)):
                    out[i][j] = min(out[i][j-1], out[i-1][j], out[i-1][j-1]) + 1
                else:
                    out[i][j] = 0
        ind = np.unravel_index(np.argmax(out, axis=None), out.shape)
        new_y, new_x = (ind)
        side = out[ind]
        result_2 = orig_img[new_y-side:new_y, new_x-side:new_x]
        crops[name] = result_2
        bbox[name] = {
				"x1": int(new_x-side),
				"x2": int(new_x),
				"y1": int(new_y-side),
				"y2": int(new_y)
		}
    
    return crops, bbox

        

        
  




