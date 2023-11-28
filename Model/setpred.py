import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel


class SetPrd(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPrd, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.bert_directory, return_dict = False)
        config = self.encoder.config
        self.num_classes = num_classes
        self.decoder = nn.Linear(config.hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask, targets=None):
        _, pooler_output = self.encoder(input_ids, attention_mask)
        class_logits = self.decoder(pooler_output)
        outputs = self.softmax(class_logits)
        best_outputs = torch.max(outputs, 1)
        if targets is not None:
            loss = F.cross_entropy(class_logits, targets.long())
            return loss, best_outputs
        else:
            return best_outputs


    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = torch.tensor([t for t in targets], dtype=torch.float, requires_grad=False).cuda()
        else:
            targets = torch.tensor([t for t in targets], dtype=torch.float, requires_grad=False)
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info


    def get_prediction(self, input_ids, attention_mask):
        with torch.no_grad():
            best_outputs = self.forward(input_ids,attention_mask)
        return best_outputs

    class SetCriterion(nn.Module):

        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, outputs, tragets):
            loss = F.cross_entropy(outputs, tragets)
            return loss