import torch 
import pandas as pd
from PIL import Image 
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms 
import pandas as pd
import matplotlib.pyplot as plt
import math

size1 = 256
size2 = 256
train_split_size = 0.3

def get_img_by_name(name):
    transform_with_size = transforms.Compose([ 
        transforms.PILToTensor(),
        transforms.Resize((size1, size2)) 
    ]) 
    transform_without_size = transforms.Compose([ 
        transforms.PILToTensor()
    ]) 
    for path in ['images/images/train/', 'images/images/test/']:
        if name in listdir(path):
            filename = join(path, name)
            if isfile(filename):
                image = Image.open(filename)
                width, height = image.size
                img_tensor_with_size = transform_with_size(image).type('torch.FloatTensor')
                img_tensor_without_size = transform_without_size(image).type('torch.FloatTensor')
                return img_tensor_with_size, img_tensor_without_size, width, height
            
def draw_img_by_name(name, model, axis, tittle = True):
    img_tensor_with_size, img_tensor_without_size, width, height = get_img_by_name(name)

    x = img_tensor_with_size
    axis.set_title(name)
    draw_img = x.type('torch.IntTensor')
    draw_points = model.forward(torch.stack([x]))[0].detach().numpy()
    axis.imshow(draw_img.permute(1,2,0))
    for j in range(0, len(draw_points),2):
                x, y = draw_points[j], draw_points[j+1] 
                axis.plot(x,y, 'ro', color = 'blue')

