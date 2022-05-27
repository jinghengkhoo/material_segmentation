import uuid

import cv2
from color import get_pixelvalues

from .createMaskJson import create_json_mask
from .get_cropped_object import get_crops
from .segment import generate_segment_image
#from .triton import material_recog_run
from .material import material_recog_run
from .triton_wood import wood_process


def material_segmentation(input_image_path, cfg, segmentation_module, class_model):

	#generate segment image
    segment_image = generate_segment_image(input_image_path, cfg, segmentation_module)

    seg_url = "static/segments/" + str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(seg_url, segment_image)
    
    # Read input image
    orig_img = cv2.imread(input_image_path)
    cv2.imwrite("input.jpg", orig_img)

    json_mask = create_json_mask(segment_image)
    crops, bbox = get_crops(json_mask, orig_img)
    
    res = []
    
    for name, crop in (crops.items()):
        cv2.imwrite(f"{name}.jpg", crop)
        confidence, label = wood_process(crop)
        material, material_conf = material_recog_run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), class_model)
        color, color_hex = get_pixelvalues(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        object_dict = {
			"bbox": bbox[name],
			"class": name,
			"attributes": {
				"color": color,
				"hex": color_hex
			}
		}
        
        if material == "Wood" or name == "floor":
            object_dict["attributes"]["type"] = label
            object_dict["attributes"]["type_confidence"] = confidence
        object_dict["attributes"]["material"] = material
        object_dict["attributes"]["material_conf"] = material_conf
        res.append(object_dict)
    
    return res, seg_url, json_mask
