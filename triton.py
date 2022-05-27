import cv2

import tritonclient.grpc as grpcclient
import numpy as np

# Classifier input size.
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONFIDENCE = 0.1

def material_recog_run(input_image):
    model_name = "material"
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

    inputs = []
    outputs = []
    # Specify the input size for classifier.
    #inputs.append(grpcclient.InferInput('input_1:0', [1, 3, INPUT_HEIGHT, INPUT_WIDTH], "FP32"))
    inputs.append(grpcclient.InferInput('x.3', [1, 3, INPUT_HEIGHT, INPUT_WIDTH], "FP32"))
    # Specify the output nodes.
    outputs.append(grpcclient.InferRequestedOutput('447'))
    mat_names = ["Cloth", "Glass", "Leather", "Others", "Plastic", "Porcelain", "Metal", "Wood"]

    #use the crop image for prediction
    # image = np.array(image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, [INPUT_WIDTH, INPUT_HEIGHT], interpolation=cv2.INTER_LINEAR)
    input_image = np.transpose(np.array(input_image, dtype=np.float32, order='C'), (2, 0, 1))
    input_image_buffer = np.expand_dims(input_image, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)
    results = triton_client.infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=None)
    
    outputs_res = results.as_numpy('447')

    output = outputs_res.reshape(-1,) # For top-k, [0.02321227 0.03176818 0.00130235 0.05652939 0.8871879].
    idx = output.argsort()[-3:][::-1] # [4 3 1].
    material = mat_names[idx[0]]
    material_conf = output[idx[0]].item()
    print(material, material_conf)
        
    return material, material_conf

if __name__ == '__main__':
    material_recog_run(cv2.imread("floor.jpg"))