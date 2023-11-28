import argparse
from utils.data import build_data
from Model.setpred import SetPrd
from trainer.trainer import Trainer



def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--generated_data_directory', type=str, default='./data/generated_data/')
    data_arg.add_argument('--dataset_name', type=str, default='cnews')
    data_arg.add_argument('--bert_directory', type=str, default='./bert-base-chinese')
    data_arg.add_argument('--train_file', type=str, default='./data/train.txt')
    data_arg.add_argument('--valid_file', type=str, default='./data/val.txt')
    data_arg.add_argument('--test_file', type=str, default='./data/test.txt')
    data_arg.add_argument('--labels_file', type=str, default='./data/class.txt')

    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default='Set_Prediction-Networks')
    learn_arg.add_argument('--batch_size', type=int, default=40)
    learn_arg.add_argument('--max_epoch', type=int, default=5)
    learn_arg.add_argument('--lr_decay', type= float, default=0.01)
    learn_arg.add_argument('--max_grad_norm', type=float, default=0)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=1e-4)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-4)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--fix_bert_embeddings' , type=str2bool, default=True)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)




    args, unparsed = get_args()
    data = build_data(args)
    num_classes = len(data.labels)
    model = SetPrd(args, num_classes)
    trainer = Trainer(model, data, args)
    trainer.train_model()