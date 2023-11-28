import copy
import os, sys
import pickle
from utils.data_process import data_process
from transformers import BertTokenizer


class Data:
    def __init__(self):
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.labels = []


    def show_data_summary(self, args):
        with open(args.labels_file) as f:
            lines = f.readlines()
        self.labels = lines
        print(self.labels)
        print("DATA SUMMARY START:")
        print('    Labels Size: %s ' % (len(self.labels)))
        print('    Train  Instance Number: %s' % (len(self.train_loader)))
        print('    Valid  Instance Number: %s' % (len(self.valid_loader)))
        print('    Test  Instance Number: %s' % (len(self.test_loader)))
        print('DATA SUMMARY END.')
        sys.stdout.flush()


    def generate_instance(self, args, data_process):
        tokenizer = BertTokenizer.from_pretrained(args.bert_directory)
        if 'train_file' in args:
            self.train_loader = data_process(args.train_file,tokenizer)
        if 'valid_file' in args:
            self.valid_loader = data_process(args.train_file,tokenizer)
        if 'test_file' in args:
            self.test_loader = data_process(args.train_file,tokenizer)


def build_data(args):

    file = args.generated_data_directory + args.dataset_name + '_' + args.model_name +'_data.pickle'
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(args)
    else:
        data = Data()
        data.generate_instance(args, data_process)
        save_data_setting(data, args)
    return data


def save_data_setting(data,args):

    new_data = copy.deepcopy(data)
    data.show_data_summary(args)
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    save_path = args.generated_data_directory + args.dataset_name + '_' + args.model_name + '_data.pickle'
    with open(save_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", save_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + '_' + args.model_name + '_data.pickle'
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary(args)
    return data