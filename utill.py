from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# 3개의 scale이 다른 feature map에서 도출되는 output의 크기를 맞춰주기 위해 사용하는 함수
# 즉, 각 feature map을 받아서, anchor를 기반으로 bounding box를 input size에 맞게 맞춰서 만드는 역할을 한다.
# 그렇게 함으로써, size가 다른 3개의 feature map의 output shape를 하나로 통일할 수 있다.
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = float(inp_dim) / float(prediction.size(2))
    grid_size = int(float(inp_dim) / stride)
    stride = int(stride)
    bbox_attr = 5 + num_classes
    num_anchors = len(anchors)
    print(prediction.shape)
    prediction = prediction.view(batch_size, bbox_attr * num_anchors, grid_size * grid_size) # 원래 feature map의 형태
    prediction = prediction.transpose(1,2).contiguous() # (batch_size, grid_size * grid_size, bbox_attr * bbox_attr)의 형태로 바꾸고 메모리에서 재배치하여 효율성을 증대시킨다. 
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attr)
    if CUDA:
        prediction = prediction.cuda()

    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors] # 원래 anchor는 input image size에 맞춰서 디자인 되어있다. 왜냐하면 최종 feature map은 input image size에 맞춰서 바뀌기 때문에
    # 굳이 계산하기 힘든 최종 feature map 기준으로 하는 것 보다, input image 기준으로 하는게 더 직관적일 뿐 더러, stride로만 나눠주면 계산도 어렵지 않으니까.

    # bounding box coordinate transformation
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # center_x의 offset
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) # center_y의 offset
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) # objectness의 probability

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_coordinate = torch.FloatTensor(a).view(-1,1)
    y_coordinate = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_coordinate = x_coordinate.cuda()
        y_coordinate = y_coordinate.cuda()

    x_y_coordinate = torch.cat((x_coordinate, y_coordinate), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0) # 모든 grid의 anchor로 부터 나온 x, y offset에 원래 x, y coordinate를 더해준다.
    prediction[:,:,:2] += x_y_coordinate

    # bounding box size transformation
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0) # anchor 3개를 grid 모양과 같이 복사해준다.
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors # 복사해준 anchor 3개를 prediction output 중 bounding box size를 의미하는 tx, ty와 곱해준다.

    # activate sigmoid function to classification score
    prediction[:,:,5 : 5 + num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])

    # transform the bounding box information to input size
    prediction[:,:,:4] *= stride

    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # The rows over threshold
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # 현재 bounding box는 center coordinate 및 width, height로 구성되어 있기 때문에 이를 4-coordinate로 변환해준다.
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # NMS Time (Image마다 가진 object의 종류가 모두 다르기 때문에, NMS를 Tensor화 해서 처리하는 것은 어렵다. 따라서 Loop를 사용해서 처리)
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # prediction shape = [grid * grid * anchor_num, bounding box information]
        # bounding box information = [bounding box center_x, center_y, w, h, objectness probability, class probability]
        max_conf, max_conf_score = torch.max(image_pred[:,5: 5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # objectness가 0인 것들을 걸러낸다.
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        try:
            image_pred = image_pred[non_zero_ind.squeeze(),:].view(-1, 7)
        except:
            continue

        if image_pred.shape[0] == 0:
            continue

        img_classes = torch.unique(image_pred[:, -1])

        # 각 class 별로 돌면서 체크한다.
        for cls in img_classes:
            # 해당 class에 해당하는 애들만 남긴다.
            cls_mask = image_pred*(image_pred[:,-1] == cls).float().unsqueeze(1)
            # 해당 class에 해당하는 애들 중에, score가 0 아닌 애들만 남긴다.
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            # 남긴 애들을 정돈한다.
            image_pred_class = image_pred[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            # 남은 애들을 objectness probability 순으로 정렬한다.
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            # 정렬한 애들을 뽑아서 새로운 리스트로 정렬한다.
            image_pred_class = image_pred_class[conf_sort_index]
            # detection의 숫자를 가져온다.
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                try:
                    ious =  bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError: # for non-optimal prediction removal
                    break
                except IndexError:
                    break

                # Zero out if the iou is less than threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                
                # Remove the zero-entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
                # make result
                batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
                seq = batch_ind, image_pred_class

                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

                try:
                    return output
                except:
                    return 0
                

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

