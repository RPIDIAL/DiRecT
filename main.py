from config import cfg, ConfigManager
import torch
from networks.network_factory import NetworkFactory
from networks.DiRecT import DiRecT

from data_loading.data_loader import CrossValidationDataLoader

from training.trainer import Trainer

from predicting.predictor import Predictor

def main():
    torch.multiprocessing.set_start_method('spawn')

    manager = ConfigManager(print_to_file = True)
    manager.start_new_training()

    cv_dataloader = CrossValidationDataLoader(cfg['data_source'], cfg['cv_fold_num'], cfg['batch_size'], cfg['test_batch_size'], num_workers=cfg['cpu_thread'])

    for cv_fold_id in range(cfg['cv_fold_num']):
        train_loader, val_loader, test_loader = cv_dataloader.get_dataloader_at_fold(cv_fold_id)
        
        factory = NetworkFactory(network = DiRecT(feature_size=3, embed_size=64, num_layers=1, num_heads=1, dim_feedforward=128, num_classes=3), device_ids = cfg['gpu'])

        model = factory.get_model()

        trainer = Trainer(model = model, train_dataloader = train_loader, val_dataloader = val_loader, learning_rate = cfg['lr'], device_ids = cfg['gpu'])
        trainer.train(epoch_num = cfg['epoch_num'], cp_filename = manager.get_checkpoint_filename_at_fold(cv_fold_id), loss_filename = manager.get_loss_filename_at_fold(cv_fold_id))
        del trainer

        model.load_state_dict(torch.load(manager.get_checkpoint_filename_at_fold(cv_fold_id))['model_state_dict'])
        predictor = Predictor(model = model, testloader = test_loader, trainloader=train_loader, device_ids = cfg['gpu'])
        predictor.predict(result_path = '{0:s}'.format(manager.test_result_path))
        del predictor

        del model
        del factory
        del train_loader, test_loader

    manager.finish_training_or_testing()

if __name__ == '__main__':
    main()