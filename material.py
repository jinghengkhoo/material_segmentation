import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import transforms

def material_recog_run(input_image, class_model):

    mat_names = ["Cloth", "Glass", "Leather", "Others", "Plastic", "Porcelain", "Metal", "Wood"]
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

    to_pil = transforms.ToPILImage()

    full_masked_pp = to_pil(input_image)

    #use the crop image for prediction
    #class_model = class_model.to(device)
    image_tensor = test_transforms(full_masked_pp).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to('cuda:0')
    output = class_model(input)
    output_numpy = output.data.cpu().numpy()
    for score in output_numpy:
        temp = np.argpartition(-score, 3)
        result_args = temp[:3]
        temp = np.partition(-score,3)
        result = -temp[:3]

    idx = np.argmax(result)
    material = mat_names[result_args[idx]]
    print("isaac raw")
    print(material, result[idx].item())
    return material, np.exp(result[idx].item())

import torch
if __name__ == "__main__":
    #init model
    CLASS_MODEL = torch.jit.load("models/traced_resnet_model.pt")
    CLASS_MODEL.eval()
    print(material_recog_run(cv2.cvtColor(cv2.imread("walnut.jpg"), cv2.COLOR_BGR2RGB), CLASS_MODEL))