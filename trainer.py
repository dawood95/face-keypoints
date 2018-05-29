import torch
import numpy as np
from tqdm import tqdm

class Trainer:

    def __init__(self, model, train_loader, val_loader,
                 cocoGt, imgIds,
                 optimizer, logger, cuda):

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.logger       = logger
        self.cuda         = cuda

        self.epoch      = 0
        self.train_iter = 0
        self.val_iter   = 0

        self.best_loss  = np.inf

    def run(self, num_epoch):
        for i in range(num_epoch):
            print("\nEpoch #%d\n"%(i))

            train_loss = self.train()
            val_loss   = self.validate()

            log_dict = {
                'train_total' : 0,
                'val_total'   : 0
            }

            for k in train_loss:
                loss = train_loss[k]
                log_dict['train_'+k] = loss
                log_dict['train_total'] += loss

            for k in val_loss:
                loss = val_loss[k]
                log_dict['val_'+k] = loss
                log_dict['val_total'] += loss

            self.logger.writer.add_scalars(
                'loss', log_dict, self.epoch
            )

            self.save(log_dict['val_total'])
            self.disp_weights()
            self.epoch += 1

    def save(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.logger.save('model_best.weights', self.model.module.state_dict())
        self.logger.save('model_last.weights', self.model.module.state_dict())

    def calc_loss(self, x, y, k, m):
        if self.cuda:
            x, y, k, m = x.cuda(), y.cuda(), k.cuda(), m.cuda()
            calc_loss = self.model.module.calc_loss
        else:
            calc_loss = self.model.calc_loss
        preds = self.model(x)
        loss  = self.model.module.calc_loss(preds, y, k, m)
        return loss # loss ---> (hm_loss, pull_loss, push_loss)

    def train(self):
        #torch.set_grad_enabled(True)
        self.model.train()

        log_loss = {
            'hm' : 0,
            'jm' : 0,
        }

        for x, y, k, m in tqdm(self.train_loader):
            hm_loss, jm_loss = self.calc_loss(x, y, k, m)

            loss = hm_loss + jm_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            hm_loss = hm_loss.item()
            jm_loss = jm_loss.item()

            if self.train_iter % 10 == 0:
                self.logger.writer.add_scalars(
                    'train',
                    {
                        'hm'  : hm_loss,
                        'jm'  : jm_loss,
                    },
                    self.train_iter
                )

            log_loss['hm'] += hm_loss
            log_loss['jm'] += jm_loss
            self.train_iter += 1

        for k in log_loss: log_loss[k] /= len(self.train_loader)
        return log_loss

    def validate(self):
        #torch.set_grad_enabled(False)
        self.model.eval()

        log_loss = {
            'hm'   : 0,
            'jm'   : 0,
        }

        for x, y, k, m in tqdm(self.val_loader):
            hm_loss, jm_loss = self.calc_loss(x, y, k, m)

            hm_loss = hm_loss.item()
            jm_loss = jm_loss.item()

            if self.val_iter % 10 == 0:
                self.logger.writer.add_scalars(
                    'val',
                    {
                        'hm' : hm_loss,
                        'jm' : jm_loss,
                    },
                    self.val_iter
                )

            log_loss['hm'] += hm_loss
            log_loss['jm'] += jm_loss

            self.val_iter += 1

        for k in log_loss: log_loss[k] /= len(self.val_loader)
        return log_loss

    def disp_weights(self):
        for name, param in self.model.named_parameters():
            self.logger.writer.add_histogram(
                name,
                param.clone().cpu().data.numpy(),
                self.epoch
            )
