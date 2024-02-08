import torch
import numpy as np
from torch import fft
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

__all__ = ['lmu_rnn_conv1d', 'slmu_rnn_ms_conv1d1']

def get_act(act_type = 'spike', **act_params):
    act_type = act_type.lower()
    if act_type == 'spike':
        return MultiStepLIFNode(**act_params, backend='cupy')
        # act_params['init_tau'] = act_params.pop('tau')
        # return MultiStepParametricLIFNode(**act_params, backend="cupy")
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'identity':
        return nn.Identity()
    
class Tokenizer(nn.Module):
    def __init__(self,
                 img_size_h=128,
                 img_size_w=128,
                 patch_size=64,
                 n_conv_layers=4,
                 in_channels=128,
                 embed_dims=256,
                 **kwargs):
        super(Tokenizer, self).__init__()
        in_planes=embed_dims
        n_filter_list = [in_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [embed_dims]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=3,
                          stride=1,
                          padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) if i == -1 else nn.Identity(),
                nn.BatchNorm1d(n_filter_list[i + 1]),
            )
                for i in range(n_conv_layers)
            ])
        
        self.rpe_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)


    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = x.permute(0,2,1).contiguous() 
        x = self.conv_layers(x)
        x_rpe = x.clone()
        x_rpe = self.rpe_conv(x_rpe)
        x = x + x_rpe
        return x

class Conv1d4EBMs(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=128, embed_dims=256, act_type='spike'):
        super().__init__()
        kernel_size = 3
        padding = 1
        groups = 1
        self.proj_conv = nn.Conv1d(in_channels, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = get_act(act_type, tau=2.0, detach_reset=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn1 = nn.BatchNorm1d(embed_dims)
        self.proj_lif1 = get_act(act_type, tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn2 = nn.BatchNorm1d(embed_dims)
        self.proj_lif2 = get_act(act_type, tau=2.0, detach_reset=True)

        self.proj_conv3 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn3 = nn.BatchNorm1d(embed_dims)
        self.proj_lif3 = get_act(act_type, tau=2.0, detach_reset=True)


        self.rpe_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        
    def forward(self, x):
        x = x.permute(0,2,1).contiguous() 
        x = self.proj_conv(x)
        # x = self.maxpool(x)
        
        x = self.proj_bn(x).permute(2,1,0).contiguous() 
        x = self.proj_lif(x).permute(2,1,0).contiguous() 
        x = self.proj_conv1(x)
        
        x = self.proj_bn1(x).permute(2,1,0).contiguous() 
        x = self.proj_lif1(x).permute(2,1,0).contiguous() 
        x = self.proj_conv2(x)

        x = self.proj_bn2(x).permute(2,1,0).contiguous() 
        x = self.proj_lif2(x).permute(2,1,0).contiguous() 
        x = self.proj_conv3(x)

        x_rpe = x.clone()
        
        x_rpe = self.proj_bn3(x_rpe).permute(2,1,0).contiguous() 
        x_rpe = self.proj_lif3(x_rpe).permute(2,1,0).contiguous() 
        x_rpe = self.rpe_conv(x_rpe) 
        x = x + x_rpe

        return x 

class LMUFFTCell(nn.Module):

    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):

        super(LMUFFTCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features = input_size, out_features = 1)
        self.f_u = nn.ReLU()

        self.W_h = nn.Linear(in_features = memory_size + input_size, out_features = hidden_size)
        self.f_h = nn.ReLU()

        
        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A)
        self.register_buffer("B", B) 

        H, fft_H = self.impulse()
        self.register_buffer("H", H) 
        self.register_buffer("fft_H", fft_H) 


    def stateSpaceMatrices(self):
        Q = np.arange(self.memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float() 
        
        return A, B


    def impulse(self):

        H = []
        A_i = torch.eye(self.memory_size)
        for t in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i

        H = torch.cat(H, dim = -1) 
        fft_H = fft.rfft(H, n = 2*self.seq_len, dim = -1)
        return H, fft_H


    def forward(self, x):

        batch_size, seq_len, input_size = x.shape
        print(x.shape, "avant le forward de la LMUFFTCELL") 
        tmp = self.W_u(x)
        print(tmp.shape)
        u = self.f_u(tmp) 

        fft_input = u.permute(0, 2, 1) 
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) 

        temp = fft_u * self.fft_H.unsqueeze(0) 

        m = fft.irfft(temp, n = 2*seq_len, dim = -1) 
        m = m[:, :, :seq_len] 
        m = m.permute(0, 2, 1) 

        input_h = torch.cat((m, x), dim = -1) 
        h = self.f_h(self.W_h(input_h)) 

        h_n = h[:, -1, :] 

        return h, h_n
    def forward_recurrent(self, x, m_last):
        u = self.f_u(self.W_u(x)) 
        m = m_last @ self.A.T + u @ self.B.T  
        input_h = torch.cat((m, x), dim = -1) 
        h = self.f_h(self.W_h(input_h)) 

        return h, m

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)
        self.fc1_dp = nn.Dropout(drop) if drop > 0. else nn.Identity()

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
        self.fc2_dp = nn.Dropout(drop) if drop > 0. else nn.Identity()
 
        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        B,C,N = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).permute(2,1,0).contiguous()
        x = self.fc1_lif(x).permute(2,1,0).contiguous()
        x = self.fc1_dp(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).permute(2,1,0).contiguous()
        x = self.fc2_lif(x).permute(2,1,0).contiguous()
        x = self.fc2_dp(x)
        return x
class ConvFFNMs(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_bn = nn.BatchNorm1d(in_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(hidden_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        B,C,N = x.shape
        x = self.fc1_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.fc1_lif(x).permute(2,1,0).contiguous()
        x = self.fc1_conv(x)
        
        
        x = x = self.fc2_bn(x).permute(2,1,0).contiguous()
        x = self.fc2_lif(x).permute(2,1,0).contiguous()
        x = self.fc2_conv(x)
        
        return x

class LMU(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = dim
        self.use_all_h = use_all_h
        self.lmu = LMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=1, theta=128)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        print(x.shape)
        x = x.transpose(-1,-2).contiguous()
        print(x.shape)
        h, h_n = self.lmu(x) 
        x = h.transpose(-1,-2).contiguous() if self.use_all_h else h_n.unsqueeze(-1) # h or h_n

        x = self.proj_conv(x)
        x = self.proj_bn(x)

        return x
    def forward_recurrent(self, x, state = None):
        batch_size = x.size(0)
        seq_len = x.size(-1)

        if state == None:
            m_0 = torch.zeros(batch_size, self.memory_size).to(x.device)
            state = m_0

        output = []
        for t in range(seq_len):
            x_t = x[:, :, t] 
            h_t, m_t = self.lmu.forward_recurrent(x_t, state)
            state = m_t
            output.append(h_t)

        output = torch.stack(output) 
        x = output.permute(1, 2, 0) if self.use_all_h else state[0].unsqueeze(-1)
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        return x 

# SPIKING
class SpikingLMUFFTCell(LMUFFTCell):

    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):

        super(SpikingLMUFFTCell, self).__init__(input_size, hidden_size, memory_size, seq_len, theta)

        if_bn = True
        self.bn_u = nn.BatchNorm1d(1) if if_bn else nn.Identity()
        self.f_u = get_act('spike', tau=2.0, detach_reset=True)
        self.bn_m = nn.BatchNorm1d(memory_size) if if_bn else nn.Identity()
        self.f_m = get_act('spike', tau=2.0, detach_reset=True)
        self.bn_h = nn.BatchNorm1d(hidden_size) if if_bn else nn.Identity()
        self.f_h = get_act('spike', tau=2.0, detach_reset=True)
    def forward(self, x):

        batch_size, seq_len, input_size = x.shape # B, N, C

        u_spike = self.f_u(self.bn_u(self.W_u(x).transpose(-1,-2)).permute(2,0,1).contiguous())

        u = u_spike.permute(1,0,2).contiguous() 

        fft_input = u.permute(0, 2, 1) 
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) 

        temp = fft_u * self.fft_H.unsqueeze(0) 

        m = fft.irfft(temp, n = 2*seq_len, dim = -1)
        m = m[:, :, :seq_len] 
        m = self.f_m(self.bn_m(m).permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
        m = m.permute(0, 2, 1)

        input_h = torch.cat((m, x), dim = -1) 

        h = self.f_h(self.bn_h(self.W_h(input_h).transpose(-1,-2)).permute(2,0,1).contiguous()) 
        h = h.permute(1,0,2).contiguous() 
        h_n = h[:, -1, :]

        h_n = h_n.unsqueeze(-1)

        return h, h_n
    def forward_recurrent(self, x, m_last):
        u_spike = self.f_u(self.bn_u(self.W_u(x).transpose(-1,-2)).permute(2,0,1).contiguous()) 
        u = u_spike.permute(1,0,2).contiguous() 
        m = m_last @ self.A.T + u @ self.B.T  
        input_h = torch.cat((m, x), dim = -1) 
        h = self.f_h(self.bn_h(self.W_h(input_h).transpose(-1,-2)).permute(2,0,1).contiguous()) 
        h = h.permute(1,0,2).contiguous()

        return h, m
class SLMUMs(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = int(dim*2.0)
        self.use_all_h = use_all_h
        self.lmu = SpikingLMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=128, theta=128) 

        self.prev_bn = nn.BatchNorm1d(dim)
        self.prev_lif = get_act('spike', tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        B, C, N = x.shape
        x = self.prev_bn(x).permute(2,1,0).contiguous() 
        x = self.prev_lif(x).permute(2,0,1).contiguous()
        h, h_n = self.lmu(x)
        x = h.transpose(-1,-2).contiguous() if self.use_all_h else h_n.unsqueeze(-1) # h or h_n


        x = self.proj_conv(x)
        x = self.proj_bn(x)

        return x
    
    def forward_recurrent(self, x, state = None):
        batch_size = x.size(0) 
        seq_len = x.size(-1)
        x = self.prev_bn(x).permute(2,1,0).contiguous() 
        x = self.prev_lif(x).permute(2,0,1).contiguous()
        if state == None:
            m_0 = torch.zeros(batch_size, self.memory_size).to(x.device).unsqueeze(1)
            state = m_0

        output = []
        for t in range(seq_len):
            x_t = x[:, 0:t+1, :] 
            h_t, m_t = self.lmu.forward_recurrent(x_t, state)
            state = torch.cat([m_0, m_t], dim=1)

        output = h_t

        x = output.permute(0, 2, 1).contiguous() if self.use_all_h else output[:,-1,:].permute(0, 2, 1).contiguous() # state is (h_n, m_n) where n = seq_len
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        return x 
# SPIKING
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, act_type='spike', attn=LMU, mlp=ConvFFN):
        super().__init__()

        self.attn = attn(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, act_type=act_type)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

### Je ne comprend pas les listes de parametres donnés en entrée ici !!
class ConvLMU(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=1, num_classes=10,
                 embed_dims=64,#[64, 128, 256], 
                 num_heads=1,#[1, 2, 4], 
                 mlp_ratios=4,#[4, 4, 4], 
                 qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.85, norm_layer=nn.LayerNorm,
                 depths=6,#[6, 8, 6], 
                 sr_ratios=8,#[8, 4, 2], 
                 T = 4, act_type='gelu', patch_embed=Tokenizer, block=Block, attn=LMU, mlp=ConvFFN, with_head_lif=False,
                 test_mode='normal'
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.with_head_lif = with_head_lif  
        self.test_mode = test_mode

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = patch_embed(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims, act_type=act_type)

        block = nn.ModuleList([block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, act_type=act_type, attn=attn, mlp=mlp)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        if self.with_head_lif:
            self.head_bn = nn.BatchNorm1d(embed_dims)
            self.head_lif = get_act(act_type, tau=2.0, detach_reset=True)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
 
        for blk in block:
            x = blk(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        
        if self.with_head_lif:
            x = self.head_bn(x)
            x = self.head_lif(x.permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
        if self.test_mode == 'all_seq':
            x = x.permute(2,0,1).contiguous() 
            x = torch.cumsum(x, dim=0)
            N, B, C = x.shape
            divisor = torch.arange(1, N + 1).view(N, 1, 1).float().to(x.device)
            x = x / divisor
        else:
            x = x.mean(dim=-1) 
        x = self.head(x) 
        return x

@register_model
def lmu_rnn_conv1d(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='relu', patch_embed=Tokenizer, attn=LMU, mlp=ConvFFN, with_head_lif=False,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def slmu_rnn_ms_conv1d1(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = ConvLMU(
        act_type='spike', patch_embed=Conv1d4EBMs, attn=SLMUMs, mlp=ConvFFNMs, with_head_lif=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model






# %%

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import PermutedMNIST
from torch.utils.data import DataLoader

# Paramètres pour l'initialisation de lmu_rnn_conv1d
embed_dims = 256  # Exemple de dimension d'embedding
num_heads = 8  # Exemple de nombre de têtes d'attention
mlp_ratio = 4.  # Exemple de ratio pour les couches MLP
depth = 6  # Exemple de profondeur de chaque bloc
num_classes = 10  # Nombre de classes dans psMNIST
img_size = 28  # Taille de l'image dans psMNIST (28x28)

# Paramètres d'entraînement
num_epochs = 50
learning_rate = 0.0001
batch_size = 1
lr_decay_factor = 0.85
lr_decay_step = 5

# Création des datasets et DataLoaders
train_dataset = PermutedMNIST(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = PermutedMNIST(train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Création du modèle
model = LMU(dim=784, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0., sr_ratio=1, use_all_h=True)  # Utilisation du modèle défini dans model.py

# Critère de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = np.reshape(images, (1, 1, 784))
        print(images.shape)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Mise à jour du taux d'apprentissage
    if (epoch + 1) % lr_decay_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_factor

    # Affichage des informations d'entraînement
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")