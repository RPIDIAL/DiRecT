import torch
import time
import os, sys
sys.path.append('../')
from loss_function.dice_loss import DiceLoss
from torch_geometric.nn import DataParallel, aggr
from tqdm import tqdm
import numpy as np

class FaceOptimizer():
    def __init__(self, model, dataloader, learning_rate, device_ids):
        self.model = model
        self.device_ids = device_ids
        self.dataloader = dataloader
        self.learning_rate = learning_rate

    def optimize_face(self, iter_num, result_path):
        os.makedirs(result_path, exist_ok=True)
        #torch.enable_grad()
        self.model.eval()

        for batch_id, batch in enumerate(self.dataloader):
            assert len(batch) == 1
            sample = batch[0]
            lmk_adj_mat = sample['lmk_adj_mat'].detach().float()
            lmk_ini = sample['lmk'].detach()
            lmk_def = torch.zeros_like(lmk_ini, requires_grad=True)
            #lmk_def = torch.rand_like(lmk_ini, requires_grad=True)
            self.optimizer = torch.optim.Adam([lmk_def], lr=self.learning_rate, weight_decay=5e-4)
            #self.optimizer = torch.optim.SGD([lmk_def], lr=self.learning_rate, momentum=0.99)
            loss_func_mandible = torch.nn.CrossEntropyLoss()
            label = torch.cat([data.y for data in batch]).to(device=sample['lmk'].device)
            gt_label = label[0, 1].detach().cpu().numpy()
            if gt_label == 0:
                continue

            final_diag = None
            for iter_id in range(iter_num):
                self.optimizer.zero_grad()                
                sample['lmk'] = lmk_ini + lmk_def
                pred = self.model(batch)
                pred_label = pred.argmax(dim=1).detach().cpu().numpy()
                optimal_target = torch.zeros_like(label, requires_grad=False).to(device=pred.device)
                diag_loss = loss_func_mandible(pred, optimal_target[:,1])
                #norm_loss = torch.sqrt((lmk_def ** 2).sum(dim=1)).mean()
                norm_loss = torch.norm(lmk_def, 2, dim=1, keepdim=False).mean()

                smth_tensor1 = lmk_adj_mat @ lmk_def
                smth_tensor2 = lmk_def
                dot_products = (smth_tensor1 * smth_tensor2).sum(dim=1)
                magnitude1 = torch.sqrt((smth_tensor1 ** 2).sum(dim=1))
                magnitude2 = torch.sqrt((smth_tensor2 ** 2).sum(dim=1))
                cosine_similarity = (dot_products + 1e-6) / (magnitude1 * magnitude2 + 1e-6)
                #smth_loss = (cosine_similarity).mean()
                smth_loss = torch.norm(smth_tensor1 - smth_tensor2, 2, dim=1, keepdim=False).mean()

                loss = diag_loss# + norm_loss + smth_loss
                print('Progress {0:5.2f}% ({1:s}) --- Iteration {2:04d}/{3:04d} --- Loss: {4:.6f} ({5:.6f}/{6:.6f}/{7:.6f}) --- GT/Pred diag (prob): {8:d}/{9:d}({10:.6f})'.format(
                        100.0 * batch_id / len(self.dataloader), sample['casename'], iter_id+1, iter_num, loss.item(), diag_loss.item(), norm_loss.item(), smth_loss.item(), int(gt_label), int(pred_label), pred[0, 0].item()))
                loss.backward()
                self.optimizer.step()
                final_diag = pred_label
            
            opt_lmk_fn = '{0:s}/{1:s}-{2:d}-{3:d}.npy'.format(result_path, sample['casename'], int(gt_label), int(final_diag))
            opt_lmk = lmk_ini + lmk_def
            opt_lmk = opt_lmk.detach().cpu().numpy()
            lmk_center = sample['lmk_center'].detach().cpu().numpy()
            lmk_std = sample['lmk_std'].detach().cpu().numpy()
            opt_lmk = opt_lmk * lmk_std + lmk_center
            np.save(opt_lmk_fn, opt_lmk)

            del batch, pred, label, optimal_target, lmk_ini, lmk_def, loss

    def save_model(self, cp_filename):
        model_cp = {}
        model_cp['model_state_dict'] = self.model.state_dict()
        model_cp['optimizer'] = self.optimizer.state_dict()
        torch.save(model_cp, cp_filename)
        print('Trained model saved to:', cp_filename)