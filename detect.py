from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utill import *
import argparse
import os
import os.path as path
from darknet import DarkNet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO V3 Model parameters')
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_avilable()
num_classes = 80
classes = load_classes('data/coco.names')

### Set up the neural model ###
print('Loading network')
model = DarkNet('cfg/yolov3.cfg')
model.load_weights('yolov3.weights')
print('The network is loaded')

model.net_info['height'] = int(args.reso)
inp_dim = int(model.net_info['height'])
assert inp_dim%32==0
assert inp_dim>32

if CUDA:
    model.cuda()

model.eval() # nn.Module을 상속받았기에, module의 기본적인 기능인 model.train()이나 model.eval() 같은 기능을 모두 사용할 수 있는 것이다.

# Read images
read_dir = time.time() # For processing time check
# Detection phase
try:
    imlist = [path.join(path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(path.join(path.realpath('.'), images))
except FileNotFoundError:
    print('No file or directory with the name {}'.format(images))
    exit()

if not os.path.exists(args.det):
    os.mkdir(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(i) for i in imlist]

# input shape을 맞춰주고, 빈 공간을 (128, 128, 128)로 채워주는 함수
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

# Forward 가능하도록 shape을 변경
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

#PyTorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    im_dim_list = im_dim_list.cuda()
    
leftover = 0
# 배치로 자르고 마지막 조금 남았을 때, 그걸 처리하기 위한 leftover 변수.
if im_dim_list % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    # batch 단위로 자르고, (), (), ... 와 같은 형식으로 concat 한 다음에, 마지막 남은 batch의 leftover까지 깔끔하게 마무리
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                    len(im_batches))]))  for i in range(num_batches)]  

### Detection time!!! ###
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # load_images
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    prediction = model(Variable(batch, volatile=True)) # volatile은 inference mode에서 사용되는 기능으로, forward pass에서는 모든 node에 관한 gradient를 모두 저장해두고 있어야하는데,
    # volatile을 True로 설정하면 이걸 저장해두지 않아서, 시간적으로나 메모리적으로나 save된다.
    prediction = write_results(prediction, confidence, num_classes, nms_thresh)

    end = time.time()

    if type(prediction) == int:
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print('{0:20s} predicted in {1:6.3f} seconds'.format(image.split('/')[-1], (end-start)/batch))
            print('{0:20s} {1:s}'.format('Objects detected:',''))
            print('-------------------------------------------')
        continue
    
    prediction[:,0] += i*batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

    try:
        output
    except NameError:
        print('No detection were made')
        exit()

    




