# @inproceedings{
#     vashishth2020compositionbased,
#     title={Composition-based Multi-Relational Graph Convolutional Networks},
#     author={Shikhar Vashishth and Soumya Sanyal and Vikram Nitin and Partha Talukdar},
#     booktitle={International Conference on Learning Representations},
#     year={2020},
#     url={https://openreview.net/forum?id=BylA_C4tPr}
# }

from utils import *
from model.layer import GGPNLayer

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p		= params
        self.act	= torch.tanh
        # self.act = torch.nn.LeakyReLU(0.2)
        self.bceloss	= torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class GGPN(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(GGPN, self).__init__(params)

        self.edge_index		= edge_index
        self.edge_type		= edge_type
        self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
        self.device		= self.edge_index.device
        self.init_rel = get_param((2*num_rel, self.p.init_dim))
        self.conv1 = GGPNLayer(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p, device=self.device)
        self.conv2 = GGPNLayer(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p, device=self.device) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent).to(self.device)))

    def forward_base(self, sub, rel, drop1, drop2):

        r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x	= drop1(x)
        x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
        x	= drop2(x) 							if self.p.gcn_layer == 2 else x
        sub_emb	= torch.index_select(x, 0, sub)
        rel_emb	= torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x

class GGPNForLinkPrediction(GGPN):
    def __init__(self, edge_index, edge_type, init_entity_embed, init_rel_embed, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0		= torch.nn.BatchNorm2d(1)
        self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
        self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
        self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
        rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp	= torch.cat([e1_embed, rel_embed], 1)
        stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):

        sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp				= self.concat(sub_emb, rel_emb)
        x				= self.bn0(stk_inp)
        x				= self.m_conv1(x)
        x				= self.bn1(x)
        x				= F.relu(x)
        x				= self.feature_drop(x)
        x				= x.view(-1, self.flat_sz)
        x				= self.fc(x)
        x				= self.hidden_drop2(x)
        x				= self.bn2(x)
        x				= F.relu(x)

        x = torch.mm(x, all_ent.transpose(1,0))
        x += self.bias.expand_as(x)
        assert not torch.isnan(x).any()
        score = torch.sigmoid(x)
        # score = x
        return score

class GGPNForEntityClassification(GGPN):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.layers = torch.nn.ModuleList()
        for l in range(self.p.hid_layer-1): 
            self.layers.append(GGPNLayer(
                self.p.gcn_dim,
                self.p.gcn_dim,
                self.p.num_rel,
                act=self.act,
                params=self.p,
                device=self.device)
            )
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.act = torch.nn.Softmax(dim = 1)
        self.w = get_param((self.p.embed_dim, self.p.num_class)).to(self.device)

    def forward(self, input=None):
        r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x	= self.drop(x)
        for layer in self.layers:
            x, r = layer(x, self.edge_index, self.edge_type, rel_embed=r)
        x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
        x = self.act(self.drop(torch.relu(x.mm(self.w))))
        return x

