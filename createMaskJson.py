#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from .src.create_annotations import *

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
#multipolygon_ids = [2, 5, 6]

# Get "images" and "annotations" info 
def images_annotations_info(mask_image, category_colors):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    #for mask_image in glob.glob(maskpath + "*.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
    #original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"
        # Open the image and (to be sure) we convert it to RGB
    mask_image_open = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image_open = Image.fromarray(mask_image_open)
    w, h = mask_image_open.size
        # "images" info 
    image = create_image_annotation(w, h, image_id)
    images.append(image)
        

    sub_masks = create_sub_masks(mask_image_open, w, h)
        
    for color, sub_mask in sub_masks.items():
        if color in category_colors.keys():
            category_id = category_colors[color]

            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)

            # Check if we have classes that are a multipolygon
            if len(polygons) > 1:
                    # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)
                                    
                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                        
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                        
                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id

def create_json_mask(mask_image, category_ids, category_colors):
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
        
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)
    
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], _ = images_annotations_info(mask_image, category_colors)

    return coco_format


if __name__ == '__main__':
    room = cv2.imread("room.png")
    coco_format = create_json_mask(room)
    with open('room.json' , "w") as outfile:
        json.dump(coco_format, outfile,indent=4)