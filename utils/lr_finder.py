import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

class ExponentialLR(_LRScheduler):

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class LR_Finder:

    def __init__(self, data, model, optim, cuda=False, name='model'):

        self.loader  = data[0]
        self.model   = model
        self.module  = model.module if cuda else model
        self.optim   = optim
        self.length  = 100
        self.sched   = ExponentialLR(optim, 1e-1, self.length)
        self.cuda    = cuda
        self.name    = name

    def run(self):
        lr = []
        losses = {
            'total' : []
        }

        self.model.train()
        calc_loss = self.module.calc_loss

        num_steps = 0
        for img, hm, mask in tqdm(self.loader):
            total_loss = {}

            if self.cuda:
                img = img.cuda(non_blocking=True)
                hm  = hm.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            preds = self.model(img)
            loss, factor = calc_loss(preds, hm, mask)

            self.sched.step()
            lr.append(self.sched.get_lr()[0])

            for k in loss:
                if k not in total_loss: total_loss[k] = 0
                total_loss[k] += loss[k]


            total_loss = 0
            for k in loss:
                total_loss += factor[k]*loss[k]
                if k not in losses: losses[k] = []
                losses[k].append(loss[k].item())

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            total_loss = total_loss.item()
            losses['total'].append(total_loss)

            num_steps +=1
            if num_steps == self.length:
                break

        for k in losses:
            plt.figure()
            plt.plot(lr, losses[k])
            plt.xlabel("Learning rate")
            plt.xscale("log")
            plt.ylabel("%s Loss"%k)
            plt.savefig("%s_%s"%(self.name, k))
