"""
Before AlexNet, SIFT(scale-invariant feature transform), SURF or HOG were some of the hand tuned feature extractors for Computer Vision.

In AlexNet, Interestingly in the lowest layers of the network, the model learned feature extractors that resembled some traditional filters.
Higher layers in the network might build upon these representations to represent larger structures, like eyes, noses, blades of grass, and so on.
Even higher layers might represent whole objects like people, airplanes, dogs, or frisbees. Ultimately, the final hidden state learns a compact
representation of the image that summarizes its contents such that data belonging to different categories can be easily separated.

Challenges perceived before AlexNet:

Computational Power:

Due to the limited memory in early GPUs, the original AlexNet used a dual data stream design, so that each of their two GPUs could be responsible
for storing and computing only its half of the model. Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models
across GPUs these days.

Data Availability:

ImageNet was released during this period by researchers under Fei-Fei Li with 1 million images, 1000 images per class with total of 1000 class.

Note:

Instead of using ImageNet, I am using MNIST and resizing the image to 224 x 224 dimension to make it justify with the AlexNet architecture.
"""
import torch
import torch.nn as nn


def euclidean_distance(x1, x2):
    if x1.shape != x2.shape:
      return 0
    dist = (x1 - x2)**2
    return 1 / (dist.sum() + 1e-8)

def cosine_similarity(x1, x2):
    x=(nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-08))
    print(x.shape)
    return sum(x)

def get_similarity(one, two):
    return euclidean_distance(one, two)

THRESHOLD=[0.5,0.5,0.5]
NUM_CACHE=4

class AlexNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.cache=[]
        self.conv1 = nn.Sequential(
            # transforming (bsize x 1 x 224 x 224) to (bsize x 96 x 54 x 54)
            #From floor((n_h - k_s + p + s)/s), floor((224 - 11 + 3 + 4) / 4) => floor(219/4) => floor(55.5) => 55
            nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3), #(batch_size * 96 * 55 * 55)
            nn.ReLU(inplace=True), #(batch_size * 96 * 55 * 55)
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)) #(batch_size * 96 * 27 * 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2), #(batch_size * 256 * 27 * 27)
            nn.ReLU(inplace=True),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)) #(batch_size * 256 * 13 * 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1), #(batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), #(batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #(batch_size * 256 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #(batch_size * 256 * 6 * 6)
            nn.Flatten()) #(batch_size * 9216)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096), #(batch_size * 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), #(batch_size * 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)) #(batch_size * 10)

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # print("self.training:{}".format(self.training))
        temp=[]
        out = self.conv1(x)
        temp.append(out)

        if not self.training and len(self.cache) > 0:
          max_,idx=0,0
          for i, cache in enumerate(self.cache):
              score = get_similarity(out, self.cache[i][0])
              print("similarity score1:{}".format(score))
              print('new net')
              if max_ < score:
                  max_=score
                  idx=i
          print("max similarity score1:{}, threshold:{}".format(max_, THRESHOLD[0]))
          if max_ > THRESHOLD[0]:
              ret=torch.autograd.Variable(self.cache[idx][-1], requires_grad=True)
              # print('return1')
              return ret

        out = self.conv2(out)
        temp.append(out)

        if not self.training and len(self.cache) > 0:
          max_,idx=0,0
          for i, cache in enumerate(self.cache):
              score = get_similarity(out, self.cache[i][1])
              print("similarity score2:{}".format(score))
              if max_ < score:
                  max_ = score
                  idx = i
          print("max similarity score2:{}, threshold:{}".format(max_, THRESHOLD[1]))
          if max_ > THRESHOLD[1]:
              ret=torch.autograd.Variable(self.cache[idx][-1], requires_grad=True)
              # print('return1')
              return ret

        out = self.conv3(out)
        temp.append(out)

        if not self.training and len(self.cache) > 0:
            max_,idx=0,0
            for i, cache in enumerate(self.cache):
                score = get_similarity(out, self.cache[i][2])
                print("similarity score3:{}".format(score))
                if max_ < score:
                    max_=score
                    idx=i
            print("max similarity score3:{}, threshold:{}".format(max_, THRESHOLD[2]))
            if max_ > THRESHOLD[2]:
                ret=torch.autograd.Variable(self.cache[idx][-1], requires_grad=True)
                # print('return1')
                return ret
        out = self.fc(out)
        temp.append(out)
        if not self.training and NUM_CACHE>0:
          self.cache.append(temp)
          print("appended to cache, shape of x:{}, len(temp):{}, len(cache):{}".format(x.shape, len(temp), len(self.cache)))
        if not self.training and len(self.cache) > NUM_CACHE:
          self.cache.pop(0)
          print("popped from cache, shape of x:{}, len(temp):{}, len(cache):{}".format(x.shape, len(temp), len(self.cache)))
        
        return out


# #GPU Usage for NVIDIA: nvidia-smi --loop=1

