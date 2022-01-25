import torch
import torch.nn as nn
from run_nerf_helpers import Embedder
from torchvision.models import resnet18


def vec2ss_matrix(vectors):  # vector to skewsym. matrix
    ss_matrices = []
    for vector in vectors:
        elements = torch.Tensor([0, -vector[2], vector[1], vector[2], 0, -vector[0], -vector[1], vector[0], 0])
        ss_matrices.append(elements.reshape(3,3))
    '''
    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]
    '''
    return torch.stack(ss_matrices)

def update_pose(start_pose, theta, v, w):
    w_skewsym = vec2ss_matrix(w)
    a = torch.eye(3) + torch.sin(theta) * w_skewsym + (
            1 - torch.cos(theta)) * torch.matmul(w_skewsym, w_skewsym)
    b = torch.matmul(torch.eye(3) * theta + (1 - torch.cos(theta)) * w_skewsym + (
            theta - torch.sin(theta)) * torch.matmul(w_skewsym, w_skewsym), v)
    ab = torch.cat((a, b.unsqueeze(1)), dim=1)
    exp_i = torch.cat((ab, torch.Tensor([0, 0, 0, 1]).unsqueeze(0)), dim=0)
    '''
            exp_i[:3, :3] = torch.eye(3) + torch.sin(theta) * w_skewsym + (
                        1 - torch.cos(theta)) * torch.matmul(w_skewsym, w_skewsym)
            exp_i[:3, 3] = torch.matmul(torch.eye(3) * theta + (1 - torch.cos(theta)) * w_skewsym + (
                        theta - torch.sin(theta)) * torch.matmul(w_skewsym, w_skewsym), v)
            exp_i[3, 3] = 1.
            '''
    est_pose = torch.matmul(exp_i, start_pose)
    return est_pose


class APR(nn.Module):
    def __init__(self):
        super(APR, self).__init__()
        # Use a pretrained VIT
        encoder = resnet18(pretrained=True)
        # Remove last layer
        self.encoder = nn.Sequential(*(list(encoder.children())[:-1]))

        self.encoder_dim = 2048
        self.fc = nn.Linear(self.encoder_dim, self.encoder_dim)
        self.w_reg = nn.Linear(self.encoder_dim // 2, 3)
        self.v_reg = nn.Linear(self.encoder_dim // 2, 3)
        self.theta_reg = nn.Linear(self.encoder_dim // 2, 1)


    def forward(self, img):
        x = self.fc(self.encoder(img))
        w = self.w_reg(x)
        v = self.v_reg(x)
        theta = self.theta_reg(x)

        start_pose = torch.eye(img.shape,4).to(img.device).to(img.dtype)
        return update_pose(start_pose, theta, v, w)


class RPR(nn.Module):
    def __init__(self):
        """
        """
        super(RPR, self).__init__()

        multires = 12
        embed_kwargs = {
            'include_input': False,
            'input_dims': 2,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        # output dim =  2 * multires * input_dims

        self.position_embedder = Embedder(**embed_kwargs)
        multires = 4
        embed_kwargs = {
            'include_input': False,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.rgb_embedder = Embedder(**embed_kwargs)

        transformer_dim = self.rgb_embedder.out_dim * 2
        self.cls_token = nn.Parameter(torch.zeros((1,transformer_dim)), requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                nhead=4,
                                                dim_feedforward=transformer_dim*4,
                                                dropout=0.1,
                                                activation="gelu")

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                      num_layers=5,
                                                      norm=nn.LayerNorm(transformer_dim))


        self.mlp = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim,  transformer_dim//2),
            nn.GELU(),
            nn.Dropout(0.1))

        self.w_reg = nn.Linear(transformer_dim // 2, 3)
        self.v_reg = nn.Linear(transformer_dim // 2, 3)
        self.theta_reg = nn.Linear(transformer_dim // 2, 1)

    def forward(self, position, sample_obs, sample_rendered, start_pose):
            # Embed and concatenate rgb
            e_sample_obs = self.rgb_embedder.embed(sample_obs)
            e_sample_rendered = self.rgb_embedder.embed(sample_rendered)
            src = torch.cat((e_sample_obs, e_sample_rendered), dim=1).unsqueeze(1) # S x 1 x D*2
            # Prepend class token
            cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
            src = torch.cat([cls_token, src])

            position = position + 1
            ext_position = torch.zeros((1, 2), device=position.device, dtype=position.dtype)
            ext_position = torch.cat((ext_position, position), dim=0)
            e_position = self.position_embedder.embed(ext_position).unsqueeze(1)

            # Add the position embedding
            src = src + e_position

            # Transformer Encoder pass
            x = self.transformer_encoder(src)[0]

            # Regress the pose correction params
            x = self.mlp(x)
            w = self.w_reg(x)
            v = self.v_reg(x)
            theta = self.theta_reg(x)

            return update_pose(start_pose, theta, v, w)

