import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tensorflow.keras.applications.resnet50 import preprocess_input

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONFIDENCE = 0.3
NMS = 0.4

def preprocess(img, input_shape):
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.dstack([img, img, img])
    img = img.astype(np.float32)
    return img

def wood_process(input_image):
    #start_time = time.time()
    model_name = "wood"
    # Create server context
    triton_client = grpcclient.InferenceServerClient(
        url='triton:8001',
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None)
    
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input_1', [1, INPUT_WIDTH, INPUT_HEIGHT, 3], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('dense_4'))

    input_image_buffer = preprocess(input_image, [INPUT_WIDTH, INPUT_HEIGHT])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    input_image_buffer = preprocess_input(input_image_buffer)
    inputs[0].set_data_from_numpy(input_image_buffer)

    results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

    result = results.as_numpy('dense_4')

    #processing_time = time.time() - start_time\

    label_file = open("labels/wood.txt").readlines()
    cate_idx2label = {}
    cate_idx2prop = {}
    for i, line in enumerate(label_file):
        cate_idx2label[i], cate_idx2prop[i] = line.strip('\n').split()

    det_id = 0

    output = result.reshape(-1,) # For top-k
    idx = output.argsort()[-3:][::-1]

    confidence = float(output[idx[0]])
    label = cate_idx2label[idx[0]]
    prop = cate_idx2prop[idx[0]]
    det_id += 1
    
    return confidence, label, prop

if __name__ == '__main__':
    print(wood_process(cv2.imread("floor.jpg")))
