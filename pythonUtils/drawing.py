import torch 
import pandas as pd
from PIL import Image 
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms 
import pandas as pd
import matplotlib.pyplot as plt
import math
import pythonUtils.config as config

def draw_pred_line(axs, dataset, model, label, num, cats_nums):
    axs[num,0].set_ylabel(label, fontsize=16)
    for i in range(cats_nums):
        x = dataset[i]
        if type(x) == tuple:
            x = x[0]
        axs[num, i].set_title(str(i))
        draw_img = x.type('torch.IntTensor')
        draw_points = model.forward(torch.stack([x]))[0].detach().numpy()
        if config.grayscale:
            axs[num, i].imshow(draw_img.permute(1,2,0), cmap = 'gray')
        else: 
            axs[num, i].imshow(draw_img.permute(1,2,0))
        for j in range(0, len(draw_points),2):
                    x, y = draw_points[j], draw_points[j+1] 
                    axs[num, i].plot(x,y, 'bo')

def get_img_by_name(name):
    transform_with_size = transforms.Compose([ 
        transforms.Resize((config.size, config.size)),
        transforms.PILToTensor()
    ]) 
    transform_without_size = transforms.Compose([ 
        transforms.PILToTensor()
    ]) 
    if config.grayscale:
        transform_with_size = transforms.Compose([transforms.Grayscale(), transform_with_size])
    if config.grayscale:
        transform_without_size = transforms.Compose([transforms.Grayscale(), transform_without_size])  
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
    csv = pd.read_csv('train_labels.csv')
    x = img_tensor_with_size
    axis.set_title(name)
    draw_img = x.type('torch.IntTensor')
    draw_points = model.forward(torch.stack([x]))[0].detach().numpy()
    if config.grayscale:
        axis.imshow(draw_img.permute(1,2,0), cmap = 'gray')
    else:
        axis.imshow(draw_img.permute(1,2,0))
    for j in range(0, len(draw_points),2):
                x, y = draw_points[j], draw_points[j+1] 
                axis.plot(x,y, 'bo')
    arr = csv.loc[csv['file_name'] == name].drop('file_name', axis = 1).values[0]
    if len(arr) > 0:
         for j in range(0, len(arr),2):
                x, y = arr[j]*(config.size/width), arr[j+1]*(config.size/height) 
                axis.plot(x,y, 'go')