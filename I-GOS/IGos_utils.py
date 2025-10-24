print("igos' utils.py file is successfully imported")
#coding=utf-8
# Generating video using I-GOS
# python Version: python3.6
# by Zhongang Qi (qiz@oregonstate.edu)
# from util import *
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2
#coding=utf-8


import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters


from typing import Tuple
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor




def topmaxPixel(deletion_HattMap: np.ndarray, thre_num: int) -> Tuple[np.ndarray, float]:
    '''
    parameters:
    -----
    HattMap: it is a deletion mask/heatmap
    thre_num:

    output:
    ------
    OutHattMap:
    img_ratio :
    '''
    import pickle
    ###bm1###
    # print("the Hattmap is ", HattMap)
    # filename = 'my_data.pkl'
    # 3. Open the file in 'wb' mode and dump the object into it.
    # with open(filename, 'wb') as file:
        # pickle.dump(HattMap, file)

    # print(f"Object has been saved to {filename}")
    flatten = deletion_HattMap.ravel()
    idx_sort_ascending = np.argsort(flatten)
    first_smallest_threnumth =idx_sort_ascending[: thre_num] #first thre
    ii = np.unravel_index(first_smallest_threnumth, deletion_HattMap.shape)

    OutHattMap = deletion_HattMap*0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    OutHattMap = 1 - OutHattMap

    #######bm1###
    # filename = 'OutHattMap.pkl'
    # # 3. Open the file in 'wb' mode and dump the object into it.
    # with open(filename, 'wb') as file:
    #     pickle.dump(OutHattMap, file)

    # # print(f"Object has been saved to {filename}")
    #####################
    return OutHattMap, img_ratio


def topmaxPixel_insertion(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    # print(ii)
    OutHattMap = HattMap * 0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size

    return OutHattMap, img_ratio







def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(\
                    img: np.ndarray,
                    use_cuda: bool = True,
                    require_grad: bool = False) -> torch.Tensor:
    """
    input: 
    - img: BGR image 
    Most pretrained PyTorch models (like ResNet, VGG) expect input images
    normalize with these.

    This reverses the channel order. If your input is in RGB,
    it becomes BGR (common when reading images with OpenCV).

    output:
    1. `preprocessed_img_tensor`: normalized
    """
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    means = [0.4915, 0.4823, 0.4468]
    stds =  [0.2470, 2435, 0.2616]




    preprocessed_img = img.copy()[:, :, ::-1]# (224,224,BGR) -> (224, 224, RGB) which is weird!

    for i in range(3):
        # But notice: this assumes img has pixel values already scaled to [0,1],
        # otherwise the normalization won’t work as intended.
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        #Ensures the array is contiguous in memory for efficient tensor conversion.

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=require_grad)


def numpy_to_torch(img, use_cuda=1, requires_grad=False):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model_new(use_cuda = 1, model_name = 'resnet50'):

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'resnet50':
        checkpoint = torch.load('./checkpoint/ckpt.pth')


    #print(model)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model


def save_heatmap(output_path, upsampled_mask, img, blurred, blur_mask=0):
    """
    input:
    save_heatmap(output_file, upsampled_mask, img * 255, blurred_img, blur_mask=0)
    """

    mask  = upsampled_mask
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
    mask = 1 - mask

    # if blur_mask:
    #     mask = cv2.GaussianBlur(mask, (11, 11), 10)
    #     mask = np.expand_dims(mask, axis=2)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255


    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    IGOS = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8)* heatmap;



    cv2.imwrite(output_path + "heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(output_path + "IGOS.png", np.uint8(255 * IGOS))
    cv2.imwrite(output_path + "blurred.png", np.uint8(255 * blurred))




def save_new(mask, img, blurred):
    ########################
    # generate the perturbed image
    #
    # parameters:
    # mask: the generated mask its the deletion mask
    # img: the original image
    # blurred: the baseline image



    #output
    # ed, ed ,ed masked image
    ####################################################
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0)) # from channels-firs torch tensor to numpy/TF/keras channel last tensor
    img = np.float32(img) / 255
    perturbated = np.multiply(mask, img) + np.multiply(1-mask, blurred)
    perturbated = cv2.cvtColor(perturbated, cv2.COLOR_BGR2RGB)
    return perturbated
