import torch
import time
import os, sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

class Predictor():
    def __init__(self, model, testloader, trainloader, device_ids):
        self.model = model
        self.device_ids = device_ids
        self.testloader = testloader
        self.trainloader = trainloader

    def predict(self, result_path, recon_enabled=False):
        os.makedirs(result_path, exist_ok=True)
        if recon_enabled:
            recon_path = "{0:s}/recon".format(result_path)
            os.makedirs(recon_path, exist_ok=True)
        torch.no_grad()
        self.model.eval()

        df_fn = '{0:s}/prediction.csv'.format(result_path)
        df = None
        if os.path.exists(df_fn):
            df = pd.read_csv(df_fn, index_col=[0])
        new_data = []

        for batch in tqdm(self.testloader, desc="Testing ... "):
            if recon_enabled:
                pred, recon, _ = self.model(batch)
            else:
                pred, recon, _ = self.model(batch)
            batch_slicer = 0
            for i, data in enumerate(batch):

                pred2_cls = torch.argmax(pred[i,:].view(-1)).detach().cpu().numpy()
                pred2_prob = pred[i,pred2_cls].detach().cpu().numpy()
                prob_0 = pred[i,0].detach().cpu().numpy()
                prob_1 = pred[i,1].detach().cpu().numpy()
                prob_2 = pred[i,2].detach().cpu().numpy()

                new_data.append([data.casename, 0, 1.0, pred2_cls, pred2_prob, prob_0, prob_1, prob_2])

                if recon_enabled:
                    recon_lmk = recon[i,:,:].detach().cpu().numpy()
                    lmk_center = data['lmk_center'].detach().cpu().numpy()
                    lmk_std = data['lmk_std'].detach().cpu().numpy()
                    recon_lmk = recon_lmk * lmk_std + lmk_center
                    recon_fn = "{0:s}/{1:s}.npy".format(recon_path, data.casename)
                    np.save(recon_fn, recon_lmk)

                batch_slicer += data.num_nodes
            del batch, pred

        new_df = pd.DataFrame(new_data, columns=['Case', 'Maxilla_Cls', 'Maxilla_Prob', 'Mandible_Cls', 'Mandible_Prob', 'Prob_0', 'Prob_1', 'Prob_2'])
        if df is None:
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(df_fn)