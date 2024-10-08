import os, sys 
sys.path.append('../')
from torch_geometric.loader import DataListLoader
import random
from data_loading.dense_lmk_dataset import GraphDataset
import pandas as pd

class CrossValidationDataLoader():
    def __init__(self, data_source_path, fold_num, batch_size, test_batch_size, num_workers):
        
        self.data_source_path = data_source_path
        self.fold_num = fold_num
        self.batch_size = batch_size
        self.test_batch_size = batch_size
        self.num_workers = num_workers
        fold_file_name = self.load_or_create_fold_file()
        self.data_list = self.load_data_list(fold_file_name)

        folds_size = [len(x) for x in self.data_list.values()]
        print('Size of folds:', folds_size)

    def load_or_create_fold_file(self):
        fold_file_name = '{0:s}/data_split-{1:d}-fold-CV.txt'.format(sys.path[0], self.fold_num)
        if not os.path.exists(fold_file_name):
            fold_entries = {}
            data_list = self.scan_data_source(self.data_source_path)
            class_p_sample_list = []
            class_r_sample_list = []
            class_n_sample_list = []
            for ele in data_list:
                if ele[3] == 'p':
                    class_p_sample_list.append(ele)
                elif ele[3] == 'r':
                    class_r_sample_list.append(ele)
                else:
                    class_n_sample_list.append(ele)
            random.shuffle(class_p_sample_list)        
            random.shuffle(class_r_sample_list)        
            random.shuffle(class_n_sample_list)        
            ptr_p = 0
            ptr_r = 0
            ptr_n = 0
            for fold_id in range(self.fold_num-1):
                fold_entries[fold_id] = \
                        class_p_sample_list[ptr_p:ptr_p+int(len(class_p_sample_list)/float(self.fold_num)+0.5)] + \
                        class_r_sample_list[ptr_r:ptr_r+int(len(class_r_sample_list)/float(self.fold_num)+0.5)] + \
                        class_n_sample_list[ptr_n:ptr_n+(len(class_n_sample_list)//self.fold_num)]
                ptr_p += int(len(class_p_sample_list)/float(self.fold_num)+0.5)
                ptr_r += int(len(class_r_sample_list)/float(self.fold_num)+0.5)
                ptr_n += len(class_n_sample_list)//self.fold_num
            fold_entries[self.fold_num-1] = class_p_sample_list[ptr_p:] + class_r_sample_list[ptr_r:] + class_n_sample_list[ptr_n:]

            with open(fold_file_name, 'w') as fold_file:
                for fold_id in range(self.fold_num):
                    for [case_name, case_path, _, _] in fold_entries[fold_id]:
                        fold_file.write('{0:d} {1:s} {2:s}\n'.format(fold_id, case_name, case_path))
        return fold_file_name
    
    def load_data_list(self, fold_file_name):
        fold_data = {}
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in fold_data:
                    fold_data[fold_id] = []
                fold_data[fold_id].append([params[1], params[2]])
        return fold_data
    
    def load_unlabeled_data_list(self, data_path):
        data_list = []
        for data_fn in os.listdir(data_path):
            if data_fn.startswith("0") and data_fn.endswith(".npy"):
                case_name = data_fn.split(".npy")[0]
                case_path = "{0:s}/{1:s}".format(data_path, data_fn)
                data_list.append([case_name, case_path])
        return data_list

    def scan_data_source(self, data_source_path):
        data_list = []
        for casename in os.listdir(data_source_path):
            if not casename.endswith('.npy'):
                continue
            if casename.startswith('CT'):
                continue
            casename = casename.split('.npy')[0]
            label_fn = "{0:s}/diagnosis.csv".format(data_source_path)
            label_df = pd.read_csv(label_fn)
            temp_df = label_df.loc[label_df['Subject ID'] == casename]
            if len(temp_df) == 0:
                print("Unlabeled case:", casename)
                continue
            case_path = '{0:s}/{1:s}.npy'.format(data_source_path, casename)

            data_list.append([casename, case_path, temp_df["Maxilla (protruded, retruded, normal)"].to_list()[0], temp_df["Mandible (protruded, retruded, normal)"].to_list()[0]])
        return data_list

    def get_dataloader_at_fold(self, test_fold_id, test_only=False):
        train_data_list, test_data_list = None, None

        test_data_list = self.data_list[test_fold_id].copy()
        for fold_id in range(self.fold_num):
            if fold_id in [test_fold_id]:
                continue
            if train_data_list is None:
                train_data_list = self.data_list[fold_id].copy()
            else:
                train_data_list.extend(self.data_list[fold_id])
                
        unlabeled_train_data_list = self.load_unlabeled_data_list("/fast/xux12/data/facial_lmk_database")

        if test_only:
            train_loader = None
            unlabeled_train_loader = None
        else:
            train_loader = DataListLoader(GraphDataset(train_data_list, enable_augmentation=True), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            unlabeled_train_loader = DataListLoader(GraphDataset(unlabeled_train_data_list, enable_augmentation=True), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataListLoader(GraphDataset(test_data_list, enable_augmentation=False), batch_size=self.test_batch_size, shuffle=False)

        return train_loader, unlabeled_train_loader, test_loader

