import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append('{}/../data_preparation'.format(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd

class GraphDataset(Dataset):

    add_class_node = False

    upper_face_landmarks = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 79, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 137, 139, 141, 142, 143, 144, 145, 147, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 166, 168, 173, 174, 188, 189, 190, 193, 195, 196, 197, 198, 203, 205, 209, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 293, 294, 295, 296, 297, 298, 299, 300, 301, 305, 309, 323, 326, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 362, 363, 366, 368, 370, 371, 372, 373, 374, 376, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 392, 398, 399, 412, 413, 414, 417, 419, 420, 423, 425, 429, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
        ]
    
    left_face_landmarks = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
    ]

    right_face_landmarks = [
        0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 94, 151, 152, 164, 168, 175, 195, 197, 199, 200, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
    ]

    exclusive_landmarks = [
        68, 104, 69, 108, 337, 299, 333, 298, # top 2nd row landmarks (excluding #151)

        10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, # outer boundary loop of landmarks

        33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, # eye (left)
        263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, # eye (right)
        
        239, 79, 166, 59, 75, 60, 20, 238, 241, 125, 242, 141, # nose bottom region (left)
        459, 309, 392, 289, 305, 290, 250, 458, 461, 354, 370, 462, # nose bottom region (right)

        62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, # lip regions
    ]

    def __init__(self, case_list, enable_augmentation=False):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.lmk_extend = []
        self.label_map = {
            'n':0,
            'r':1,
            'p':2,
            }
        self.enable_augmentation = enable_augmentation
        self.valid_landmark_ids, _, self.adj_matrix, self.adj_edge, self.symmetric_pairs, self.upper_face_lmk_ids, self.lower_face_lmk_ids, self.left_face_lmk_ids, self.right_face_lmk_ids = self.load_facial_template_model()
        self.data_list = []
        for [casename, case_path] in tqdm (case_list, desc="Loading data ..."):
            graph_data = self.load_case_data(casename, case_path)
            self.data_list.append(graph_data)

        self.lmk_extend = np.array(self.lmk_extend)
        np.min(self.lmk_extend[:, [0,2,4]], axis=0)
        np.max(self.lmk_extend[:, [1,3,5]], axis=0)
        print("Landmark extend:", 
              np.min(self.lmk_extend[:, [0,2,4]], axis=0),
              np.max(self.lmk_extend[:, [1,3,5]], axis=0),
              np.quantile(self.lmk_extend[:, [0,2,4]], 0.50, axis=0),
              np.quantile(self.lmk_extend[:, [1,3,5]], 0.50, axis=0))

    def len(self):
        return len(self.data_list)
    
    def load_facial_template_model(self):
        obj_file_path = f"{os.path.dirname(os.path.realpath(__file__))}/canonical_face_model.obj"

        # Lists to store vertex coordinates and face definitions
        vertices = []
        faces = []
        symmetric_dict = {}
        with open(obj_file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    # Add vertex coordinates, converting each to float
                    vertices.append([float(coord) for coord in parts[1:4]])
                    symmetric_dict["{0:s} {1:s} {2:s}".format(parts[1], parts[2], parts[3])] = len(vertices) - 1
                elif parts[0] == 'f':
                    # Add face definitions, converting indices to int and subtracting 1 for 0-based indexing
                    faces.append([int(index.split('/')[0]) - 1 for index in parts[1:]])
        
        # Add a class node (like class token in Transformer)
        if GraphDataset.add_class_node:
            vertices.append([0.0, 0.0, 0.0])

        # Convert the vertices list to a NumPy array
        vertex_array = np.array(vertices)

        # Initialize the adjacency matrix with zeros
        N = len(vertices)
        adjacency_matrix = np.zeros((N, N), dtype=int)
        symmetric_matrix = np.zeros((N, N), dtype=int)

        # Fill the adjacency matrix based on face definitions
        for face in faces:
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    adjacency_matrix[face[i], face[j]] = 1
                    adjacency_matrix[face[j], face[i]] = 1  # Ensure the matrix is symmetric

        # Class node is connected with all the other nodes
        if GraphDataset.add_class_node:
            adjacency_matrix[-1, :] = 1
            adjacency_matrix[:, -1] = 1
        
        # Fill the symmetric matrix
        for key in symmetric_dict:
            sym_key = "-{0:s}".format(key)
            if sym_key in symmetric_dict:
                symmetric_matrix[symmetric_dict[key], symmetric_dict[sym_key]] = 1
                symmetric_matrix[symmetric_dict[sym_key], symmetric_dict[key]] = 1

        # Exclude some specified landmarks
        valid_landmark_ids = []
        for i in range(N):
            if i not in GraphDataset.exclusive_landmarks:
                valid_landmark_ids.append(i)
        vertex_array = vertex_array[valid_landmark_ids, :]
        adjacency_matrix = adjacency_matrix[valid_landmark_ids, :]
        adjacency_matrix = adjacency_matrix[:, valid_landmark_ids]
        symmetric_matrix = symmetric_matrix[valid_landmark_ids, :]
        symmetric_matrix = symmetric_matrix[:, valid_landmark_ids]
        
        symmetric_pairs = []
        for i in range(symmetric_matrix.shape[0]):
            for j in range(i + 1, symmetric_matrix.shape[1]):
                if symmetric_matrix[i, j] == 1:
                    symmetric_pairs.append([i, j])
        symmetric_pairs = np.array(symmetric_pairs)
        
        edge_start_nodes, edge_end_nodes = np.nonzero(adjacency_matrix)
        edge_index = np.concatenate([edge_start_nodes.reshape(1,-1), edge_end_nodes.reshape(1,-1)], axis=0)

        upper_face_lmk_ids = []
        lower_face_lmk_ids = []
        left_face_lmk_ids = []
        right_face_lmk_ids = []
        for lmk_new_id, valid_lmk_id in enumerate(valid_landmark_ids):
            if valid_lmk_id in GraphDataset.upper_face_landmarks:
                upper_face_lmk_ids.append(lmk_new_id)
            else:
                lower_face_lmk_ids.append(lmk_new_id)
            if valid_lmk_id in GraphDataset.left_face_landmarks:
                left_face_lmk_ids.append(lmk_new_id)
            if valid_lmk_id in GraphDataset.right_face_landmarks:
                right_face_lmk_ids.append(lmk_new_id)

        print("Upper face landmark #:", len(upper_face_lmk_ids))
        print("Lower face landmark #:", len(lower_face_lmk_ids))
        print("Left face landmark #:", len(left_face_lmk_ids))
        print("Right face landmark #:", len(right_face_lmk_ids))
        print("Valid landmark #:", len(valid_landmark_ids))
        print("Exclude landmarks:", GraphDataset.exclusive_landmarks)

        return valid_landmark_ids, vertex_array, adjacency_matrix, edge_index, symmetric_pairs, upper_face_lmk_ids, lower_face_lmk_ids, left_face_lmk_ids, right_face_lmk_ids
    
    def load_case_data(self, casename, case_path):
        lmk_fn = case_path
        label_fn = "/data/facial_lmk_database/diagnosis.csv" # path to ground-truth diagnosis file

        lmk_array = np.load(lmk_fn)

        norm_lmk_array = lmk_array - lmk_array[4,:]
        self.lmk_extend.append([
            np.nanmin(norm_lmk_array[:,0]), 
            np.nanmax(norm_lmk_array[:,0]),
            np.nanmin(norm_lmk_array[:,1]), 
            np.nanmax(norm_lmk_array[:,1]),
            np.nanmin(norm_lmk_array[:,2]), 
            np.nanmax(norm_lmk_array[:,2]),
            ])
        
        point_center = lmk_array[4,:].copy() # use Landmark #4 as the original point (0, 0, 0)
        if GraphDataset.add_class_node:
            lmk_array = lmk_array[self.valid_landmark_ids[:-1], :] # exclude the specified landmarks
        else:
            lmk_array = lmk_array[self.valid_landmark_ids, :] # exclude the specified landmarks
        invalid_lmk_ids = np.isnan(lmk_array).any(axis=1)
        if invalid_lmk_ids.sum() > 0:
            for invalid_lmk_id in np.nonzero(invalid_lmk_ids)[0]:
                print("Invalid landmark detected:", self.valid_landmark_ids[invalid_lmk_id])
            assert invalid_lmk_ids.sum() == 0, "Invalid landmark (NaN) detected in case: {0:s}".format(casename) # make sure there is no invalid
        point_std = np.std(lmk_array, axis=0, keepdims=True)
        point_std[:] = 100.0
        lmk_array = (lmk_array - point_center) / point_std
        if GraphDataset.add_class_node:
            lmk_array = np.append(lmk_array, [[0.0, 0.0, 0.0]], axis=0)
        lmk_tensor = torch.tensor(lmk_array, dtype=torch.float)
        lmk_center = torch.tensor(point_center, dtype=torch.float)
        lmk_std = torch.tensor(point_std, dtype=torch.float)
        
        x = lmk_tensor.detach()
        mesh_edge_index = torch.tensor(self.adj_edge, dtype=torch.long)
        mesh_edge_weight = torch.ones(self.adj_edge.shape[1], dtype=torch.long)
        lmk_adj_mat = torch.from_numpy(self.adj_matrix)

        label_df = pd.read_csv(label_fn)
        temp_df = label_df.loc[label_df['Subject ID'] == casename]
        maxilla_label = temp_df['Maxilla (protruded, retruded, normal)'].to_list()[0]
        mandible_label = temp_df['Mandible (protruded, retruded, normal)'].to_list()[0]
        y = torch.zeros(1, 2, dtype=torch.long)
        y[0,0] = self.label_map[maxilla_label]
        y[0,1] = self.label_map[mandible_label]
        batch = torch.zeros(x.shape[0]).long()

        upper_face_lmk_ids = torch.tensor(self.upper_face_lmk_ids, dtype=torch.long)
        lower_face_lmk_ids = torch.tensor(self.lower_face_lmk_ids, dtype=torch.long)
        left_face_lmk_ids = torch.tensor(self.left_face_lmk_ids, dtype=torch.long)
        right_face_lmk_ids = torch.tensor(self.right_face_lmk_ids, dtype=torch.long)
        unmasked_face_lmk_ids = torch.arange(len(self.valid_landmark_ids), dtype=torch.long)

        graph_data = Data(
            x=x, 
            edge_index=mesh_edge_index, 
            edge_weight=mesh_edge_weight,
            lmk=lmk_tensor,
            lmk_center=lmk_center,
            lmk_std=lmk_std,
            lmk_adj_mat=lmk_adj_mat,
            upper_face_lmk_ids=upper_face_lmk_ids,
            lower_face_lmk_ids=lower_face_lmk_ids,
            left_face_lmk_ids=left_face_lmk_ids,
            right_face_lmk_ids=right_face_lmk_ids,
            unmasked_face_lmk_ids=unmasked_face_lmk_ids,
            casename=casename, 
            y=y, 
            sample_idx=-1,
            batch=batch)

        return graph_data
    
    def augment_graph(self, graph_data, scale_max, mirroring_prob):        
        lmk_array = graph_data.lmk.detach()
        lmk_std = graph_data.lmk_std.detach()

        yaw_max, pitch_max, roll_max = 0.00, 0.00, 0.00
        yaw_angle = (torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)*2-1) * yaw_max # in radian unit
        pitch_angle = (torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)*2-1) * pitch_max # in radian unit
        roll_angle = (torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)*2-1) * roll_max # in radian unit
        mirroring_p = torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)
        mask_prob = torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)

        sin_yaw = torch.sin(yaw_angle)
        cos_yaw = torch.cos(yaw_angle)
        sin_pitch = torch.sin(pitch_angle)
        cos_pitch = torch.cos(pitch_angle)
        sin_roll = torch.sin(roll_angle)
        cos_roll = torch.cos(roll_angle)

        yaw_mat = torch.tensor([[cos_yaw, -sin_yaw, 0],[sin_yaw, cos_yaw, 0],[0, 0, 1]])
        pitch_mat = torch.tensor([[cos_pitch, 0, sin_pitch],[0, 1, 0],[-sin_pitch, 0, cos_pitch]])
        roll_mat = torch.tensor([[1, 0, 0],[0, cos_roll, -sin_roll],[0, sin_roll, cos_roll]])
        trans_mat = yaw_mat @ pitch_mat @ roll_mat

        lmk_array = (trans_mat @ lmk_array.transpose(1, 0)).transpose(1, 0)

        rescale_factor = 1.0 + (torch.rand(1, dtype=lmk_array.dtype, device=lmk_array.device)*2-1) * scale_max
        lmk_array = lmk_array * rescale_factor

        if mirroring_p > (1 - mirroring_prob):
            lmk_array[:, 0] = -lmk_array[:, 0]
            tmp_array = lmk_array[self.symmetric_pairs[:,1], :].detach()
            lmk_array[self.symmetric_pairs[:,1], :] = lmk_array[self.symmetric_pairs[:,0], :].detach()
            lmk_array[self.symmetric_pairs[:,0], :] = tmp_array

        graph_data.lmk = lmk_array
        graph_data.x = lmk_array.detach()
        graph_data.lmk_std = lmk_std

        return graph_data
    
    def mask_out_graph(self, graph_data):
        masked_graph_data = self.data_list[graph_data.sample_idx].detach()
        masked_graph_data.sample_idx = graph_data.sample_idx
        masked_graph_data = self.augment_graph(masked_graph_data, scale_max=0.01, mirroring_prob=0.5)

        return masked_graph_data
    
    def get(self, idx):
        graph_data = self.data_list[idx].detach()
        graph_data.sample_idx = idx

        if self.enable_augmentation:
            graph_data = self.augment_graph(graph_data, scale_max=0.00, mirroring_prob=0.0)

        return graph_data