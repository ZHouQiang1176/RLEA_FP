import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from os.path import dirname, abspath
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


def build_graph():
    path1 = dirname(dirname(abspath(__file__))) + '/data/edges_1.dat'
    path2 = dirname(dirname(abspath(__file__))) + '/data/edges_2.dat'
    f = open(path1, "r")
    for line in f:
        x1 = eval(line)

    f1 = open(path2, "r")
    for line in f1:
        x2 = eval(line)
    g = dgl.DGLGraph()
    g.add_nodes(710)
    g.add_edges(x1, x2)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(x2, x1)

    return g


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.action_space = action_space
        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = torch.distributions.Categorical

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, states, deterministic=False):
        state = copy.deepcopy(states)
        features = state['Node_feature'] #[n, macro_num, 8]
        index = torch.tensor(state['Current_macro_id']).cuda()
        obs = state['Obs']
        obs = torch.tensor(obs).cuda().unsqueeze(1)
        features = torch.tensor(features).cuda()
        mask = state['Mask']
        mask = torch.tensor(mask).cuda()
        shape_mask = state["Shape_Mask"]
        shape_mask = torch.tensor(shape_mask).cuda()
        index_0 = torch.arange(0,len(index)).cuda()
        index = torch.stack([index_0,index]).tolist()

        value, logits, next_shape, action = self.base(obs, features, index, mask, shape_mask)
        dist_act = self.dist(logits)
        dist_shape = self.dist(next_shape)
        shape = dist_shape.sample()

        action_log_prob = dist_act.log_prob(action)
        shape_log_prob = dist_shape.log_prob(shape)
        
        action =torch.concat([action.unsqueeze(1),shape.unsqueeze(1)],dim=1)
        return value, action, action_log_prob+shape_log_prob
        
    def get_value(self, states):
        state = copy.deepcopy(states)
        features = state['Node_feature'] #[n, macro_num, 8]
        index = torch.tensor(state['Current_macro_id']).cuda()
        obs = state['Obs']
        obs = torch.tensor(obs).cuda().unsqueeze(1)
        features = torch.tensor(features).cuda()
        mask = state['Mask']
        mask = torch.tensor(mask).cuda()
        shape_mask = state["Shape_Mask"]
        shape_mask = torch.tensor(shape_mask).cuda()
        index_0 = torch.arange(0,len(index)).cuda()
        index = torch.stack([index_0,index]).tolist()

        value, _, _, _ = self.base(obs, features, index, mask, shape_mask)
        return value

    def evaluate_actions(self, state_batch, action):
        #输入是一个batch数据
        features = [state['Node_feature'] for state in state_batch]
        features = np.array(features)
        features = torch.tensor(features).cuda()
        index = [state['Current_macro_id'] for state in state_batch]
        mask = [torch.tensor(state['Mask']) for state in state_batch]
        shape_mask = [torch.tensor(state['Shape_Mask']) for state in state_batch]
        mask = torch.stack(mask).cuda()
        shape_mask = torch.stack(shape_mask).cuda()

        index_0 = torch.arange(0,len(state_batch)).cuda()
        index = torch.tensor(index).cuda()
        index = torch.stack([index_0,index]).tolist()
        obs = [state['Obs'] for state in state_batch]
        obs  = torch.stack(obs).unsqueeze(1).cuda()

        value, logits,next_shape, _ = self.base(obs,features,index,mask, shape_mask,False,action[:,0])
        dist = self.dist(logits)
        dist_shape = self.dist(next_shape)

        shape_log_probs = dist_shape.log_prob(action[:,1])
        action_log_probs = dist.log_prob(action[:,0])
        dist_entropy = dist.entropy().mean()
        shape_entropy = dist_shape.entropy().mean()

        return value, action_log_probs+shape_log_probs, dist_entropy,shape_entropy

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x,0), nn.init.calculate_gain('relu'))
        #
        # 128
        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(32 * 12 * 12, hidden_size - 16)), nn.ReLU())

        #256 :[1,1,256,256]—>[1,16,64,64]->[1,32,32,32]->[1,16,16,16]->[1,8,14,14]
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 8, 8, stride=4, padding=2)), nn.LeakyReLU(),
            init_(nn.Conv2d(8, 16, 4, stride=2,padding=1)), nn.LeakyReLU(),
            init_(nn.Conv2d(16, 16, 4, stride=2,padding=1)), nn.LeakyReLU(),
            init_(nn.Conv2d(16, 8, 3, stride=1)), nn.LeakyReLU(), Flatten(),
            init_(nn.Linear(8 * 14 * 14, hidden_size - 256)), nn.LeakyReLU())
            #init_(nn.Linear(7200, hidden_size - 256)), nn.LeakyReLU())
        

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x,0))
        self.mlp_layer = nn.Sequential(init_(nn.Linear(8,32)), nn.ReLU(),init_(nn.Linear(32,256)), nn.LeakyReLU())
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.shape_actor = init_(nn.Linear(hidden_size+1, 543))
        self.actor_net = init_(nn.Linear(hidden_size, 256*256))
        self.train()
        self.distr = torch.distributions.Categorical

    def forward(self, obs, features, index, mask, shape_mask, flag=False,action=None):
        x = self.main(obs)
        if flag:
            x4 = self.mlp_layer(features)[0][index].unsqueeze(0)
        else:
            x4 = self.mlp_layer(features)[index] 
        embedding_feature = torch.cat([x, x4], 1)

        act_logits = self.actor_net(embedding_feature)
        act_min = torch.ones_like(mask)*(-2**32+1)
        act_logits = torch.where(mask>0,act_logits,act_min.cuda()).squeeze()
        act_logits = F.softmax(act_logits,-1)
        dist_act = self.distr(act_logits)
        if action is None:
            action = dist_act.sample()

        act_embedding = (action/(256*256)).unsqueeze(1)
        shape_logits = self.shape_actor(torch.concat((embedding_feature,act_embedding),dim=1))
        shape_min = torch.ones_like(shape_mask)*(-2**32+1)
        shape_logits = torch.where(shape_mask>0,shape_logits,shape_min.cuda()).squeeze()

        next_shape = F.softmax(shape_logits,-1)

        value = self.critic_linear(embedding_feature)
        return value, act_logits, next_shape, action