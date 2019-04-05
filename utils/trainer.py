import torch
import numpy as np
from torch import nn

class Trainer:

    def __init__(self, data, model, scheduler, optimizer, logger, cuda=False):

        self.train_loader = data[0]
        self.val_loader   = data[1]

        self.model  = model
        self.module = model.module if cuda else model
        self.sched  = scheduler
        self.optim  = optimizer
        self.logger = logger
        self.cuda   = cuda

        self.epoch      = 0
        self.train_step = 0
        self.val_step   = 0

        self.best_loss = np.inf

    def run(self, num_epoch):
        for i in range(num_epoch):
            self.sched.step()

            print("\nEPOCH #%d\n"%(self.epoch+1))

            with self.logger.comet.train():
                train_loss = self.train()

            with self.logger.comet.validate():
                val_loss   = self.validate()

            self.logger.comet.log_multiple_metrics({
                'train': train_loss,
                'val'  : val_loss
            }, prefix='total_loss', step=(self.epoch+1))

            self.save(val_loss)
            self.epoch += 1

    def save(self, loss):
        data = {
            'state_dict': self.module.state_dict(),
            'optim'     : self.optim.state
        }

        if loss < self.best_loss:
            self.best_loss = loss
            self.logger.save('model_best_loss.weights', data)

        self.logger.save('model_%d.weights'%(self.epoch+1), data)

    def train(self):
        self.model.train()
        calc_loss = self.module.calc_loss

        num_steps  = 0
        total_loss = 0

        for img, hm in self.train_loader:
            if self.cuda:
                img = img.cuda(non_blocking=True)
                hm  = hm.cuda(non_blocking=True)

            preds = self.model(img)
            loss, factor = calc_loss(preds, hm)

            step_loss = 0
            for k in _loss:
                step_loss += _loss[k]*factor[k]

            if self.train_step % 10 == 0:
                total = step_loss.item()
                print("Train %d :"%num_steps, end='')
                for k in _loss:
                    self.logger.comet.log_metric(k, _loss[k].item(), self.train_step)
                    print(" %s=[%.5f]"%(k, _loss[k].item()), end='')
                self.logger.comet.log_metric("total", total, self.train_step)
                print(" total=[%.5f]"%(total))

            self.optim.zero_grad()
            step_loss.backward()
            self.optim.step()

            total_loss += step_loss.item()
            self.train_step += 1
            num_steps  += 1

        total_loss /= num_steps

        print("Train : Total Loss=[%.5f]"%(total_loss))

        return total_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        calc_loss = self.module.calc_loss

        num_steps  = 0
        total_loss = 0

        for img, hm in self.val_loader:
            if self.cuda:
                img = img.cuda(non_blocking=True)
                hm  = hm.cuda(non_blocking=True)

            preds = self.model(img)
            _loss, factor = calc_loss(preds, hm)

            loss  = 0
            for k in _loss:
                loss += _loss[k]*factor[k]

            if self.val_step % 10 == 0:
                total = loss.item()
                print("Val %d :"%num_steps, end='')

                for k in _loss:
                    self.logger.comet.log_metric(k, _loss[k].item(), self.val_step)
                    print(" %s=[%.5f]"%(k, _loss[k].item()), end='')
                self.logger.comet.log_metric("total", total, self.val_step)
                print(" total=[%.5f]"%(total))

            total_loss += loss.item()
            self.val_step += 1
            num_steps  += 1

        total_loss /= num_steps

        print("Val : Total Loss=[%.5f]"%(total_loss))

        return total_loss
