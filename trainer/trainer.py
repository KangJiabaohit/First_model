import gc
import random
import torch.nn.utils
from torch import nn, optim
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.metric import metric


class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if component[1] in n],
                'lr': args.decoder_lr
            },
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()


    def train_model(self):
        best_precision = 0
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size +1
        for epoch in range(self.args.max_epoch):
            #Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                if not train_instance:
                    continue
                input_ids, attention_mask, targets, _ = self.model.batchify(train_instance)
                loss, _ = self.model(input_ids, attention_mask, targets)
                avg_loss.update(loss.item(), 1)
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1 ) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 5 == 0 and batch_id !=0:
                    print("  Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            print("=== Epoch %d Validation ===" % epoch)
            precision =self.eval_model(self.data.valid_loader, self.data.labels)
            if precision > best_precision:
                print("Achieving Best Result on Validation Set", flush=True)
                best_precision = precision
                best_result_epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on validatoin set is %f achieving at epoch %d." % (best_precision, best_result_epoch), flush=True)


    def eval_model(self,eval_loader, labels):
        self.model.eval()
        prediction = []
        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, targets, _ = self.model.batchify(eval_instance)
                prediction = self.model.get_prediction(input_ids, attention_mask)
        return metric(prediction, targets, labels)


    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        if epoch !=0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
        return optimizer