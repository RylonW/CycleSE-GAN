import random
import time
import datetime
import sys
import shutil

from torch.autograd import Variable
import torch
#from visdom import Visdom
import numpy as np
from matplotlib import pyplot as plt

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        #self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                #self.losses[loss_name] = losses[loss_name].data[0]
                self.losses[loss_name] = losses[loss_name].item()
            else:
                #self.losses[loss_name] += losses[loss_name].data[0]
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        #for image_name, tensor in images.items():
            #if image_name not in self.image_windows:
                #self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            #else:
                #self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                #if loss_name not in self.loss_windows:
                    #self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    #opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                #else:
                    #self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# 初始化权重正态分布
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #torch.nn.init.normal(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
# 绘制loss图像
# 因visdom无法使用，所以Logger里的Draw images完全用不了
class loss_plot():
    def __init__(self, epoch):
        self.epoch = epoch
        self.totalloss = np.zeros(epoch)
        self.loss_D_A = np.zeros(epoch)
        self.loss_D_B = np.zeros(epoch)
        
    def accumulate_loss(self, epoch, loss_G, loss_D_A, loss_D_B):
        self.totalloss[epoch] += loss_G
        self.loss_D_A[epoch] += loss_D_A
        self.loss_D_B[epoch] += loss_D_B
    
    def plot_loss(self, now_epoch):
        if(now_epoch == self.epoch - 1):
            x = np.arange(self.epoch)

            # loss_G
            plt.figure(0)
            plt.plot(x, self.totalloss)
            plt.title("G_loss")
            plt.grid(True) 
            # save the figure
            plt.savefig("G_loss.png")
            shutil.move("G_loss.png","./output")

            # loss_D_A
            plt.figure(1)
            plt.plot(x, self.loss_D_A)
            plt.title("D_A_loss")
            plt.grid(True)
            plt.savefig("D_A_loss.png")
            shutil.move("D_A_loss.png","./output")

            # loss_D_B
            plt.figure(2)
            plt.plot(x, self.loss_D_B)
            plt.title("D_B_loss")
            plt.grid(True)
            plt.savefig("D_B_loss.png")
            shutil.move("D_B_loss.png","./output")
            