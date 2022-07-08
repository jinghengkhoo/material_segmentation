import uuid

import cv2
from color import get_pixelvalues

from .createMaskJson import create_json_mask
from .get_cropped_object import get_crops
from .segment import generate_segment_image
#from .triton import material_recog_run
from .material import material_recog_run
from .triton_wood import wood_process


def material_segmentation(input_image_path, cfg, segmentation_module, use_second_gpu):

	#generate segment image
    segment_image = generate_segment_image(input_image_path, cfg, segmentation_module, use_second_gpu)

    seg_url = "static/segments/" + str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(seg_url, segment_image)
    
    # Read input image
    orig_img = cv2.imread(input_image_path)
    cv2.imwrite("input.jpg", orig_img)

    # Label ids of the dataset
    category_ids = {
        "outlier": 0,
        "floor": 1,
        "wall": 2,
        "ceiling": 3,
        "rug": 4, 
    }

    # Define which colors match which categories in the images
    category_colors = {
        "(0, 0, 0)": 0, # Outlier
        "(80, 50, 50)": 1, # floor
        "(120, 120, 120)": 2, # Wall
        "(120, 120, 80)": 3, # cieling
        "(255, 9, 92)": 4, # rug
    }

    json_mask = create_json_mask(segment_image, category_ids, category_colors)
    crops, bbox = get_crops(json_mask, orig_img)

    category_names = {}
    segments = {}

    for key, value in category_ids.items():
        category_names[value] = key

    for annot in json_mask["annotations"]:
        segments[category_names[annot["category_id"]]] = annot["segmentation"]
    
    res = []
    
    for name, crop in (crops.items()):
        filename = f'static/objects/{str(uuid.uuid4())}.png'
        cv2.imwrite(filename, crop)
        confidence, label, prop = wood_process(crop)
        material, material_conf = material_recog_run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        color, color_hex = get_pixelvalues(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        object_dict = {
			"bbox": bbox[name],
			"class": name,
			"attributes": {
				"color": color,
				"hex": color_hex
			},
            "segmentation": segments[name],
            "crop": filename
		}
        
        if material == "Wood" or name == "floor":
            object_dict["attributes"]["type"] = label
            object_dict["attributes"]["type_confidence"] = confidence
            object_dict["attributes"]["property"] = prop
        object_dict["attributes"]["material"] = material
        object_dict["attributes"]["material_conf"] = material_conf
        res.append(object_dict)
    
    return res, seg_url
