from __future__ import division # python 2에서 사용되는 library

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules import module, padding
from torch.nn.modules.container import ModuleList
from utill import *

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

# parse한 block을 기반으로 module을 만들고
# 만들어진 module을 조합해서 forward를 진행하는것이 바로 이 DarkNet class
class DarkNet(nn.Module):
    def __init__(self, cfg_file):
        super().__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:] # 0은 net_info
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])
            print(module_type)
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(i) for i in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                # transformation
                x = x.data # 의미 파악
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                
                else:
                    detections = torch.cat((detections, x), 1) # concatenate for each feature maps

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')

        # The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

def get_test_output():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalize
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

        

model = DarkNet("cfg/yolov3.cfg")
inp = get_test_output()
pred = model(inp, torch.cuda.is_available())
print(pred)
model = DarkNet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")

            
            


            
    



    