cfg = {}

cfg['gpu'] = [6,7] # to use multiple gpu: cfg['gpu'] = [0,1,2,3]
cfg['cv_fold_num'] = 4
cfg['batch_size'] = 32
cfg['test_batch_size'] = 2
cfg['lr'] = 0.001
cfg['cpu_thread'] = 0 # multi-thread for data loading. zero means single thread.
cfg['epoch_num'] = 800
cfg['model_path'] = '/fast/xux12/proj/facial_deform_diag/models'
cfg['data_source'] = '/fast/xux12/data/facial_lmk_database'

import os, sys
import time

class ConfigManager():
    def __init__(self, print_to_file = True):
        self.print_to_file = print_to_file

    def start_new_training(self):
        self.start_time = time.localtime()
        time_stamp = time.strftime("%Y-%m-%d_%H%M%S", self.start_time)
        
        # create directory for results storage
        self.store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
        os.makedirs(self.store_dir, exist_ok=True)
        if self.print_to_file:
            cmd_output_fn = '{}/cmd_output.txt'.format(self.store_dir)
            self.cmd_output_file = open(cmd_output_fn, 'a')
            sys.stdout = self.cmd_output_file
            sys.stderr = self.cmd_output_file
        self.loss_filename = '{}/loss.txt'.format(self.store_dir)
        self.config_filename = '{}/config.txt'.format(self.store_dir)
        self.save_configuration()
        self.test_result_path = '{}/results_test'.format(self.store_dir)
        os.makedirs(self.test_result_path, exist_ok=True)
        self.cache_path = '{}/cache'.format(self.store_dir)
        os.makedirs(self.cache_path, exist_ok=True)

        print("Start time: {start_time}\n".format(start_time=time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)))
    
    def start_from_checkpoint(self, model_name):
        self.start_time = time.localtime()
        
        # create directory for results storage
        self.store_dir = '{}/{}'.format(cfg['model_path'], model_name)
        if self.print_to_file:
            cmd_output_fn = '{}/cmd_output.txt'.format(self.store_dir)
            self.cmd_output_file = open(cmd_output_fn, 'a')
            sys.stdout = self.cmd_output_file
            sys.stderr = self.cmd_output_file
        self.loss_filename = '{}/loss.txt'.format(self.store_dir)
        self.config_filename = '{}/config.txt'.format(self.store_dir)
        self.save_configuration()
        self.test_result_path = '{}/results_test'.format(self.store_dir)
        os.makedirs(self.test_result_path, exist_ok=True)
        self.opt_face_result_path = '{}/results_opt_face'.format(self.store_dir)
        os.makedirs(self.opt_face_result_path, exist_ok=True)
        self.cache_path = '{}/cache'.format(self.store_dir)
        os.makedirs(self.cache_path, exist_ok=True)

        print("Start time: {start_time}\n".format(start_time=time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)))

    def get_checkpoint_filename_at_fold(self, fold_id):
        cp_filename = '{0:s}/cp_cv_{1:d}.pth.tar'.format(self.store_dir, fold_id)
        return cp_filename

    def get_loss_filename_at_fold(self, fold_id):
        cp_filename = '{0:s}/loss_cv_{1:d}.txt'.format(self.store_dir, fold_id)
        return cp_filename

    def finish_training_or_testing(self):
        self.finish_time = time.localtime()
        print("Finish time: {finish_time}\n".format(finish_time=time.strftime("%Y-%m-%d %H:%M:%S", self.finish_time)))
        cost_secs = time.mktime(self.finish_time) - time.mktime(self.start_time)
        print("Time cost: {0:>02d}:{1:>02d}:{2:>02d}\n\n".format(int(cost_secs) // 3600, (int(cost_secs) % 3600) // 60, int(cost_secs) % 60))
        self.remove_cache()

    def save_configuration(self):
        print("Saving configurations to file:", self.config_filename)
        print()
        lines = "Configuration:\n"
        for cfg_key in cfg:
            lines += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
        with open(self.config_filename, 'w') as config_file:
            config_file.write(lines)
        print(lines)

    def remove_cache(self):
        if 'cache' in self.cache_path and os.path.exists(self.cache_path):
            for fn in os.listdir(self.cache_path):
                if fn.endswith('.npz'):
                    os.remove('{0:s}/{1:s}'.format(self.cache_path, fn))
            os.rmdir(self.cache_path)

    def start_testing(self):
        self.start_time = time.localtime()
        time_stamp = cfg['test_model_name'][6:]
        
        # create directory for results storage
        self.store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
        if self.print_to_file:
            cmd_output_fn = '{}/cmd_test_output.txt'.format(self.store_dir)
            self.cmd_output_file = open(cmd_output_fn, 'w')
            sys.stdout = self.cmd_output_file
            sys.stderr = self.cmd_output_file
        self.test_result_path = '{}/results_test'.format(self.store_dir)
        self.opt_face_result_path = '{}/results_opt_face'.format(self.store_dir)

        print("Start time: {start_time}\n".format(start_time=time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)))
