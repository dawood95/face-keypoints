import json
import torch
import numpy as np
from tqdm import tqdm
from utils.eval import hm2kpts
from pycocotools.cocoeval import COCOeval

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

        self.best_acc  = 0
        self.cocoGt    = cocoGt
        self.imgIds    = imgIds

    def run(self, num_epoch):
        for i in range(num_epoch):
            print("\nEpoch #%d\n"%(i))

            train_loss = self.train()
            val_loss, val_acc = self.validate()

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

            self.save(val_acc)
            self.disp_weights()
            self.epoch += 1

    def save(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
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
        return (*loss, preds) # loss ---> (hm_loss, pull_loss, push_loss)

    def train(self):
        #torch.set_grad_enabled(True)
        self.model.train()

        log_loss = {
            'hm' : 0,
            'jm' : 0,
        }

        for x, y, k, m, img_id in tqdm(self.train_loader):
            hm_loss, jm_loss, _ = self.calc_loss(x, y, k, m)

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

        result = []
        for x, y, k, m, img_ids in tqdm(self.val_loader):
            hm_loss, jm_loss, preds = self.calc_loss(x, y, k, m)

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

            # record acc
            hm_preds, jm_preds = preds[-1]
            for i in range(hm_preds.shape[0]):
                result.extend(
                    hm2kpts(
                        hm_preds[i].detach().cpu().numpy(),
                        jm_preds[i].detach().cpu().numpy(),
                        img_ids[i].item(),
                        scale=(x.shape[3] / hm_preds.shape[3], x.shape[2] / hm_preds.shape[2])
                    )
                )
            
        for k in log_loss: log_loss[k] /= len(self.val_loader)

        stats = [0]
        if len(result) > 0:
            with open('tmp.json', 'w') as outFile:
                json.dump(result, outFile)
            cocoDt = self.cocoGt.loadRes('tmp.json')
            cocoEval = COCOeval(self.cocoGt, cocoDt, 'keypoints')
            cocoEval.params.imgIds = self.imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            stats = cocoEval.stats
            logger_prefix=''
            self.logger.writer.add_scalars(
                'mAP',
                {
                    logger_prefix+'0.5:0.95:all':  stats[0],
                    logger_prefix+'0.5:all': stats[1],
                    logger_prefix+'0.75:all': stats[2],
                    logger_prefix+'0.5:0.95:medium': stats[3],
                    logger_prefix+'0.5:0.95:large': stats[4],
                },
                self.epoch
            )

            self.logger.writer.add_scalars(
                'mAR',
                {
                    logger_prefix+'0.5:0.95:all': stats[5],
                    logger_prefix+'0.5:all': stats[6],
                    logger_prefix+'0.75:all': stats[7],
                    logger_prefix+'0.5:0.95:medium': stats[8],
                    logger_prefix+'x0.5:0.95:large': stats[9],
                },
                self.epoch
            )

        return log_loss, stats[0]

    def disp_weights(self):
        for name, param in self.model.named_parameters():
            self.logger.writer.add_histogram(
                name,
                param.clone().cpu().data.numpy(),
                self.epoch
            )
