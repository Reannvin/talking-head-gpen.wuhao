import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from .model import BiSeNet
import torchvision.transforms as transforms

class FaceParsing():
    def __init__(self):
        self.net = self.model_init()
        self.preprocess = self.image_preprocess()

    def model_init(self, 
                   resnet_path='./checkpoint/resnet18-5c106cde.pth', 
                   model_pth='./checkpoint/79999_iter.pth'):
        net = BiSeNet(resnet_path)
        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth)) 
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        net.eval()
        return net

    def image_preprocess(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image, size=(512, 512)):
        if isinstance(image, str):
            image = Image.open(image)

        with torch.no_grad():
            image = image.resize(size, Image.BILINEAR)
            img = self.preprocess(image)
            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()
            else:
                img = torch.unsqueeze(img, 0)
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            parsing[np.where(parsing > 14)] = 0
            gray_mask = parsing == 14

            gray_coords = np.where(gray_mask)
            if gray_coords[0].size > 0:
                min_y = int(np.min(gray_coords[0]))
                max_y = int(np.max(gray_coords[0]))
                min_x = int(np.min(gray_coords[1]))
                max_x = int(np.max(gray_coords[1]))

                mid_start = min_x + (max_x - min_x) // 4
                mid_end = min_x + 3 * (max_x - min_x) // 4

                mid_y = (min_y + max_y)//2

                vertical_gradient = np.linspace(128, 0, max_y - mid_y + 1).astype(np.uint8)

                for y in range(min_y, max_y + 1):
                    for x in range(mid_start, mid_end + 1):
                        if gray_mask[y, x]:
                            if y < mid_y:
                                vert_value = 255
                            else:
                                vert_value = vertical_gradient[y - mid_y]
                            horiz_value = 255                                
                            parsing[y, x] = min(vert_value, horiz_value)

            for i in range(1,14):
                parsing[np.where(parsing == i)] = 255
            
        parsing = Image.fromarray(parsing.astype(np.uint8))
        return parsing


if __name__ == "__main__":
    fp = FaceParsing()
    segmap = fp('154_small.png')
    segmap.save('res.png')

    
