import torch
import time
import sys
sys.path.append('../')
from torch.optim.lr_scheduler import LambdaLR
from itertools import cycle
import copy

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, learning_rate, device_ids):
        self.student_model = model
        self.teacher_model = copy.deepcopy(model)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.device_ids = device_ids
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = torch.optim.Adam(self.student_model.module.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.ema_alpha = 0.99

    def update_teacher(self):
        with torch.no_grad():  # Ensure gradients are not computed for this operation
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data.mul_(self.ema_alpha).add_(student_param.data, alpha=1 - self.ema_alpha)

    def train(self, epoch_num, cp_filename, loss_filename):

        train_step_per_epoch = len(self.train_dataloader)

        # Configuration
        warmup_steps = 150
        total_steps = train_step_per_epoch * epoch_num
        # Lambda function for warmup
        warmup_lambda = lambda step: min(1.0, step / warmup_steps)
        # Lambda function for decay (Cosine decay after warmup)
        decay_lambda = lambda step: 1 - step / total_steps
        # Combined lambda for warmup and decay
        combined_lambda = lambda step: warmup_lambda(step) * decay_lambda(step)
        # Scheduler
        scheduler = LambdaLR(self.optimizer, lr_lambda=combined_lambda)

        torch.enable_grad()
        
        min_epoch_loss = 1e6

        loss_diag = None
        loss_reco = torch.nn.MSELoss()
        
        for epoch_id in range(epoch_num):
            epoch_t0 = time.perf_counter()
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_diag_loss = 0
            epoch_reco_loss = 0
            epoch_cons_loss = 0
            epoch_diag_loss_count = 0
            epoch_reco_loss_count = 0
            epoch_cons_loss_count = 0
            self.teacher_model.eval()
            self.student_model.train()
            if self.val_dataloader is not None:
                dataloader = zip(self.train_dataloader, cycle(self.val_dataloader))
            else:
                dataloader = self.train_dataloader
            for batch_id, batch in enumerate(dataloader):
                if self.val_dataloader is not None:
                    l_batch, ul_batch = batch[0], batch[1]
                    batch = l_batch + ul_batch
                    masked_l_batch = [self.train_dataloader.dataset.mask_out_graph(data) for data in l_batch]
                    masked_ul_batch = [self.val_dataloader.dataset.mask_out_graph(data) for data in ul_batch]
                    masked_batch = masked_l_batch + masked_ul_batch
                    labeled_sample_num = len(l_batch)
                    unlabeled_sample_num = len(ul_batch)
                    sample_num = labeled_sample_num + unlabeled_sample_num
                else:
                    l_batch = batch
                    masked_batch = [self.train_dataloader.dataset.mask_out_graph(data) for data in l_batch]
                    labeled_sample_num = len(l_batch)
                    unlabeled_sample_num = 0
                    sample_num = labeled_sample_num + unlabeled_sample_num
                self.optimizer.zero_grad()
                if unlabeled_sample_num > 0:
                    _, _, t_cls_tokens = self.teacher_model(ul_batch)
                s_pred, s_recon, s_cls_tokens = self.student_model(masked_batch)
                label = torch.cat([data.y for data in l_batch]).to(device=s_pred.device)
                recon_label = torch.cat([data.x.unsqueeze(0) for data in masked_batch]).to(device=s_pred.device)
                if loss_diag is None:
                    mandible_weight = 1.0 / torch.tensor([19, 38, 44], dtype=s_pred.dtype).to(device=s_pred.device)
                    mandible_weight = mandible_weight / mandible_weight.sum()
                    loss_diag = torch.nn.CrossEntropyLoss(weight=mandible_weight)
                diag_loss = loss_diag(s_pred[:labeled_sample_num], label[:,1])
                reco_loss = loss_reco(s_recon, recon_label)
                loss = diag_loss + 0.0 * reco_loss

                epoch_diag_loss += diag_loss.item() * labeled_sample_num
                epoch_diag_loss_count += labeled_sample_num
                epoch_reco_loss += reco_loss.item() * sample_num
                epoch_reco_loss_count += sample_num

                if unlabeled_sample_num > 0:
                    cons_loss = 1.0 - torch.nn.functional.cosine_similarity(s_cls_tokens[labeled_sample_num:,:], t_cls_tokens, dim=1).mean()
                    loss += (float(epoch_id+1) / epoch_num) * cons_loss 
                    epoch_cons_loss += cons_loss.item() * unlabeled_sample_num
                    epoch_cons_loss_count += unlabeled_sample_num
                print('Epoch {0:04d}/{1:04d} --- Progress {2:5.2f}% --- Loss: {3:.6f}({4:.6f}/{5:.6f}/{6:.6f})'.format(
                        epoch_id+1, epoch_num, 100.0 * batch_id / train_step_per_epoch, loss.item(), diag_loss.item(), reco_loss.item(), cons_loss.item()))
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                self.update_teacher()

                del batch, masked_batch, s_pred, s_recon, s_cls_tokens, t_cls_tokens, label, recon_label, loss
            epoch_diag_loss /= epoch_diag_loss_count
            epoch_reco_loss /= epoch_reco_loss_count
            if epoch_cons_loss_count > 0:
                epoch_cons_loss /= epoch_cons_loss_count

            epoch_t1 = time.perf_counter()
            epoch_t = epoch_t1 - epoch_t0

            loss_line = 'Epoch {0:04d}/{1:04d} --- Diag. loss: {2:.6f} --- Reco. loss: {3:.6f} --- Cons. loss: {4:.6f} --- time: {5:>02d}:{6:>02d}:{7:>02d} --- Lr: {8:.6f}'.format(epoch_id+1, epoch_num, epoch_diag_loss, epoch_reco_loss, epoch_cons_loss, int(epoch_t) // 3600, (int(epoch_t) % 3600) // 60, int(epoch_t) % 60, current_lr)
            if (epoch_id == 0) or (epoch_diag_loss < min_epoch_loss):
                min_epoch_loss = epoch_diag_loss
                self.save_model(cp_filename)
                loss_line += ' (optimal model saved)'
            loss_line += '\n'
            print(loss_line)
            with open(loss_filename, 'a') as loss_file:
                loss_file.write(loss_line)

        return self.student_model

    def save_model(self, cp_filename):
        model_cp = {}
        model_cp['student_model_state_dict'] = self.student_model.state_dict()
        model_cp['model_state_dict'] = self.teacher_model.state_dict()
        model_cp['optimizer'] = self.optimizer.state_dict()
        torch.save(model_cp, cp_filename)
        print('Trained model saved to:', cp_filename)