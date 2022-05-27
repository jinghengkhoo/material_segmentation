import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tensorflow.keras.applications.resnet50 import preprocess_input

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONFIDENCE = 0.3
NMS = 0.4

def preprocess(img, input_shape, transpose=(2, 0, 1), letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    #img /= 255.0
    return img

def wood_process(input_image):

    #start_time = time.time()
    model_name = "wood"
    res = {"errors": [], "success": False}
    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url='triton:8001',
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        res["errors"].append("Triton connection error")
        return res

    # Health check
    if not triton_client.is_server_live():
        res["errors"].append("Triton server not live")
        return res

    if not triton_client.is_server_ready():
        res["errors"].append("Triton server not ready")
        return res
    
    if not triton_client.is_model_ready(model_name):
        res["errors"].append(model_name + "model not ready")
        return res
    
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input_1', [1, INPUT_WIDTH, INPUT_HEIGHT, 3], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('dense_2'))

    #print("Creating buffer from image file...")
    #input_image = cv2.imread(str(path))
    if input_image is None:
        res["errors"].append(f"FAILED: could not load input image")
        return res
    input_image_buffer = preprocess(input_image, [INPUT_WIDTH, INPUT_HEIGHT], transpose=(0, 1, 2))
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    input_image_buffer = preprocess_input(input_image_buffer)
    inputs[0].set_data_from_numpy(input_image_buffer)

    #print("Invoking inference...")
    results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

    #print(f"Received result buffer of size {result.shape}")
    #print(f"Naive buffer sum: {np.sum(result)}")

    result = results.as_numpy('dense_2')
    #print(f"Detected objects: {len(detected_objects)}")

    #processing_time = time.time() - start_time

    labels_path = ["labels/wood.txt"]
    all_labels = []

    for label_file in labels_path:
        f = open(label_file, 'r')
        l = f.readlines()
        f.close()
        for t in l:
            all_labels.append(t.split("\n")[0])

    res = {"results": [], "success": True}
    det_id = 0

    res = []

    output = result.reshape(-1,) # For top-k
    idx = output.argsort()[-3:][::-1]

    confidence = float(output[idx[0]])
    label = all_labels[idx[0]]
    det_id += 1
    
    return confidence, label

if __name__ == '__main__':
    print(wood_process(cv2.imread("floor.jpg")))
