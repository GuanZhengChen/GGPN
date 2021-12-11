from utils import *
from model.message_passing import MessagePassing
import torchsnooper
from model.rff import RFF


class GGPNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None, device='cpu'):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = device

        self.w_mat = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (in_channels, in_channels, in_channels)), 
                                    dtype=torch.float, device=self.device, requires_grad=True))

        self.w_loop = get_param((in_channels, out_channels), mode='kaiming').to(self.device)
        self.w_in = get_param((in_channels, out_channels), mode='kaiming').to(self.device)
        self.w_out = get_param((in_channels, out_channels), mode='kaiming').to(self.device)

        self.w_i_loop = get_param((in_channels, in_channels), mode='kaiming').to(self.device)
        self.w_i_in = get_param((in_channels, in_channels), mode='kaiming').to(self.device)
        self.w_i_out = get_param((in_channels, in_channels), mode='kaiming').to(self.device)

        self.w_j_loop = get_param((in_channels, in_channels), mode='kaiming').to(self.device)
        self.w_j_in = get_param((in_channels, in_channels), mode='kaiming').to(self.device)
        self.w_j_out = get_param((in_channels, in_channels), mode='kaiming').to(self.device)

        self.w_map = get_param((in_channels, in_channels)).to(self.device)
        self.w_rel = get_param((in_channels, out_channels), mode='kaiming').to(self.device)

        self.loop_rel = get_param((1, in_channels), mode='kaiming')
        # self.loop_rel = get_param((self.p.num_ent, in_channels)).to(self.device)

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        # self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.rff = RFF(self.p.rff_samples // 2, self.in_channels, device=self.device)
        # self.rff = rff
        if self.p.bias:
            self.register_parameter('bias', Parameter(
                torch.zeros(out_channels).to(self.device)))
    # @torchsnooper.snoop()

    def forward(self, x, edge_index, edge_type, rel_embed):
        """
            edge_index:[(head entity_id, tail entity_id)] 前num_edges个是原始实体，后num_edges个是逆关系
            edge_type:[rel_id]
        """

        # if self.device is None:
        #     self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel],
                              dim=0).to(self.device)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:,
                                                   :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type = edge_type[:
                                                 num_edges], 	 edge_type[num_edges:]
        self.loop_index = torch.stack(
            [torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)
        # self.loop_type = torch.arange(self.num_rels, self.num_rels + num_ent, dtype=torch.long).to(self.device)
       
        # print(self.loop_type[0], self.loop_type[-1])

        # norm需要理解
        self.in_norm = self.compute_norm(self.in_index,  num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        # self.loop_norm = self.compute_norm(self.loop_index, num_ent)

        in_res = self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,
                                rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type,
                                  rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,
                                 rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out = (1/3)*self.drop(in_res) + (1/3) * self.drop(out_res) + (1/3)*loop_res
        # out = self.drop(in_res) * self.drop(out_res)
        # print(out.shape)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)

        # Ignoring the self loop inserted
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    # @torchsnooper.snoop()
    def information_extractor(self, ent_embed, rel_embed):
        trans_embed = ent_embed * rel_embed
        return trans_embed

    def rbf_kernel(self, x1, x2, sigma):
        X12norm = torch.sum((x1 - x2)**2, 1, keepdims=True)
        return torch.exp(-X12norm/(2*sigma**2))

    # @torchsnooper.snoop()
    def message(self, x_i, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode)).to(
            self.device)  # get weight by mode
        w_i = getattr(self, 'w_i_{}'.format(mode)).to(self.device)
        w_j = getattr(self, 'w_j_{}'.format(mode)).to(self.device)

        rel_emb = torch.index_select(rel_embed, 0, edge_type).to(self.device)

        xi_rel = self.information_extractor(x_i, rel_emb).mm(w_i)
        xj_rel = self.information_extractor(x_j, rel_emb).mm(w_j)


        k = self.rff(xi_rel, xj_rel)
        k = k.view(-1, 1)
        k = k.div(np.sqrt(self.in_channels)).to(self.device)
        # print(torch.min(k), torch.max(k))
        k = torch.exp(k).to(self.device)

        value = xj_rel.mm(self.w_map)
        out = torch.mm(value, weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)
        return k, out

    def update(self, aggr_out):
        return aggr_out

    # @torchsnooper.snoop()
    def multiply_turker(self, ent_embed, rel_embed):
        W_mat = torch.mm(rel_embed, self.w_mat.view(self.in_channels, -1))
        W_mat = W_mat.view(-1, self.in_channels, self.in_channels)
        W_mat = self.drop(W_mat)
        x = ent_embed.view(-1, 1, self.in_channels)
        x = torch.bmm(x, W_mat) 
        x = x.view(-1, self.in_channels) 
        return x

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float().to(self.device)
        # Summing number of weights of the edges
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
