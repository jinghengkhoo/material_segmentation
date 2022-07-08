import cv2
import numpy as np
import tritonclient.grpc as grpcclient

INPUT_HEIGHT = 224
INPUT_WIDTH = 224

def preprocess(img, input_shape):
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = img.astype(np.float32)
    return img

def material_recog_run(input_image):
    """
    Run material recognition model
    input_image is opencv RGB image
    returns material name and confidence item
    """

    mat_names = ['Bricks', 'Fabric', 'Foliage', 'Glass', 'Leather', 'Metal', 'Paper', 'Plastic', 'Stone', 'Water', 'Wood']

    #start_time = time.time()
    model_name = "material"
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
    outputs.append(grpcclient.InferRequestedOutput('dense_2'))


    input_image_buffer = preprocess(input_image, [INPUT_WIDTH, INPUT_HEIGHT])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)
    prediction = results.as_numpy('dense_2')

    prediction = prediction.reshape(-1,)
    id = np.argmax(prediction)
    conf = prediction[id].item()
    material = mat_names[id]
    return material, conf
