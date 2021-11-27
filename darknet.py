from __future__ import division # python 2에서 사용되는 library

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules import padding
from torch.nn.modules.container import ModuleList

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

# cfg file을 입력으로, 파싱을 통해 network의 모든 block을 dictionary 형태로 저장하여 key-value 값 형태로 불러올 수 있게 해주는 함수.
def parse_cfg(cfg_file):
    file = open(cfg_file, 'r')
    lines = file.read().split('\n') # 비어있는거, 주석, white space 제거해보기
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.lstrip().rstrip() for x in lines ]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[': # 새로운 block이면
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

    ## 즉, 
    ## type : convolutional
    ## pad : 56
    ## size : 224
    ## 이런 식으로 구성되어 있을 것이다. 즉, 하나의 block은 하나의 dictionary이다.

############### YOLO 내부의 5종류의 레이어의 module 제작 ##############

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 # 각 layer의 input 중 channel은, 이전 layer의 filter 수이므로, 추적이 필요하다. 처음은 image의 channel이 3이니까, 3을 대입해야한다.
    output_filters = [] # Route layer나 skip-connection layer를 위해 현 layer의 output 또한 추적한다.
    
    # blocks 내의 각 block에 대하여 module (block 및 activation function들) 형태로 만들기 위해서 nn.Sequential을 사용할 것이다.
    # 한 block은 convolution block을 예제로 들면, convolution, padding, activation function까지 여러개의 layer로 구성되어 있어서 sequential 형태가 꼭 필요하다.
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation']
            try: # 왜 두 조건이 연관된 조건도 아닌데, 하나의 try문으로 묶은거지? 4가지 경우 중 2가지밖에 고려 못할텐데.
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Module에 conv layer를 추가한다.
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module('conv_{0}'.format(index),conv)

            # Module에 batch_normalization layer를 추가한다.
            if batch_normalize:
                bn = nn.BatchNorm2d(filters) # conv layer의 채널 수와 당연히 같아야겠지. 2d짜리가 몇 개 있어야 하냐 그거잖아.
                module.add_module('batch_norm_{0}'.format(index), bn)

            # Module에 activation function을 추가한다.
            if activation == 'leaky':
                activation_function = nn.LeakyReLU(0.1, inplace = True) # inplace는 해당 function을 바로 객체에 적용하겠는가? 하는 안전장치이다.
                # 만약 inplace를 false로 해놨으면, x = activation_function(x)와 같이 해야했을거고, 아니라면 activation_function(x)와 같이 되었겠지.
                module.add_module('leaky_{0}'.format(index), activation_function)

        # Upsample layer 였을 경우
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('up_sample_{}'.format(index), upsample)

        # Route layer 였을 경우
        elif x['type'] == 'route':
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        # Shortcut layer 였을 경우
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        # YOLO layer 였을 경우
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = list(map(int, anchors))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))


            


            
    



    