import torch
import torch.nn as nn
import numpy as np
import torch.jit as jit


class MatrixLayer(jit.ScriptModule):
    __constants__ = ['input_size', 'output_size']

    def __init__(self, input_size, output_size, bias=True, cuda=True):
        super(MatrixLayer, self).__init__()

        # self.matrix_layer = nn.Parameter(torch.Tensor(1, input_size, output_size))
        # self.bias_layer = nn.Parameter(torch.Tensor(1, output_size))
        self.matrix_layer = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias_layer = nn.Parameter(torch.Tensor(1, output_size))
        # if cuda:
        #    self.matrix_layer = self.matrix_layer.cuda()
        #    if bias:
        #        self.bias_layer = self.bias_layer.cuda()
        self.init_layers()

    @jit.script_method
    def forward(self, input_matrix):
        # out = torch.bmm(input_matrix.unsqueeze_(0), self.matrix_layer)
        # out = out.squeeze_(0) + self.bias_layer
        out = torch.mm(input_matrix, self.matrix_layer) + self.bias_layer
        return out

    def init_layers(self):
        nn.init.xavier_uniform_(self.matrix_layer)
        nn.init.xavier_uniform_(self.bias_layer)


class EmbeddingLayer(jit.ScriptModule):
    __constants__ = ['input_size', 'output_size']

    def __init__(self, input_size, output_size, bias=True, cuda=True):
        super(EmbeddingLayer, self).__init__()

        # self.matrix_layer = nn.Parameter(torch.Tensor(1, input_size, output_size))
        # self.bias_layer = nn.Parameter(torch.Tensor(1, output_size))
        self.matrix_layer = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias_layer = nn.Parameter(torch.Tensor(1, output_size))
        # if cuda:
        #    self.matrix_layer = self.matrix_layer.cuda()
        #    if bias:
        #        self.bias_layer = self.bias_layer.cuda()
        self.init_layers()

    @jit.script_method
    def forward(self, index_emb):
        # out = torch.bmm(input_matrix.unsqueeze_(0), self.matrix_layer)
        # out = out.squeeze_(0) + self.bias_layer
        out = self.matrix_layer[index_emb] + self.bias_layer
        return out

    def init_layers(self):
        nn.init.xavier_uniform_(self.matrix_layer)
        nn.init.xavier_uniform_(self.bias_layer)


class ParallelGRU(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        super(ParallelGRU, self).__init__()
        self.Wxh = nn.Parameter(torch.Tensor(input_size, 3 * hidden_size))
        self.Whh = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        self.bxh = nn.Parameter(torch.Tensor(1, 3 * hidden_size))
        self.bhh = nn.Parameter(torch.Tensor(1, 3 * hidden_size))
        self.init_layers()

    def init_layers(self):
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)
        nn.init.xavier_uniform_(self.bxh)
        nn.init.xavier_uniform_(self.bhh)

    @jit.script_method
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gate_x = torch.mm(x, self.Wxh) + self.bxh
        gate_h = torch.mm(hidden, self.Whh) + self.bhh
        i_r, i_i, i_h = gate_x.chunk(3, 1)
        h_r, h_i, h_h = gate_h.chunk(3, 1)
        reset_g = torch.sigmoid(i_r + h_r)
        input_g = torch.sigmoid(i_i + h_i)
        h_tilde = torch.tanh(i_h + (reset_g * h_h))

        hy = h_tilde + input_g * (hidden - h_tilde)

        return hy.squeeze_()

    def adv_forward(self, x: torch.Tensor, hidden: torch.Tensor, noise) -> torch.Tensor:
        adv_noise_wxh = noise['Wxh']
        adv_noise_whh = noise['Whh']
        adv_noise_bxh = noise['bxh']
        adv_noise_bhh = noise['bhh']
        gate_x = torch.mm(x, self.Wxh + adv_noise_wxh) + (self.bxh + adv_noise_bxh)
        gate_h = torch.mm(hidden, self.Whh + adv_noise_whh) + (self.bhh + adv_noise_bhh)
        i_r, i_i, i_h = gate_x.chunk(3, 1)
        h_r, h_i, h_h = gate_h.chunk(3, 1)
        reset_g = torch.sigmoid(i_r + h_r)
        input_g = torch.sigmoid(i_i + h_i)
        h_tilde = torch.tanh(i_h + (reset_g * h_h))

        hy = h_tilde + input_g * (hidden - h_tilde)

        return hy.squeeze_()


class SessionGRU(jit.ScriptModule):
    __constants__ = ['session_layers']

    def __init__(self, input_size, hidden_size, output_size, session_layers=[100],
                 dropout_hidden=.5, dropout_input=0, batch_size=50, use_cuda=True):

        super(SessionGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.session_layers = session_layers
        self.dropout_input = dropout_input
        self.dropout_p_hidden = dropout_hidden

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.input_to_session = MatrixLayer(input_size, hidden_size)
        self.hidden_to_output = MatrixLayer(hidden_size, output_size)
        self.final_activation = nn.Tanh()
        self.hidden_activation = nn.Tanh()
        self.dropout_hidden = nn.Dropout(p=self.dropout_p_hidden)
        self.gru = ParallelGRU(input_size=hidden_size, hidden_size=hidden_size)
        self.onehot_buffer = self.init_emb()
        self.embedding_input = EmbeddingLayer(self.input_size, self.hidden_size)
        self = self.to(self.device)

    def forward_(self, input, hidden):

        embedded = self.onehot_encode(input)
        h_s = self.input_to_session(embedded)

        Hs_new = []
        for i in range(len(self.session_layers)):
            session_in = torch.mm(h_s, self.Ws_in[i]) + self.Bs_h
            rz_in, h_in = session_in[self.session_layers[0]:],

            # if rz_noise is not None:
            #    rz_u = torch.sigmoid(rz_noise + session_in[self.session_layers[0]:] + (torch.mm(hidden[i], self.Ws_rz[i])).t())
            # else:
            rz_u = torch.sigmoid(session_in[self.session_layers[0]:] + (torch.mm(hidden[i], self.Ws_rz[i])).t())

            h_s = self.hidden_activation(torch.mm(hidden[i] * rz_u[:self.session_layers[0]].t(), self.Ws_hh[i]).t()
                                         + session_in[:self.session_layers[0]])
            z = rz_u[self.session_layers[0]:].t()
            h_s = (1.0 - z) * hidden[i] + z * h_s.t()
            h_s = self.dropout_hidden(h_s)
            Hs_new.append(h_s)

        hidden = torch.stack(Hs_new)

        # output = h_s.view(-1, h_s.size(-1)) # (B,H)
        # if torch.isnan(h_s.max()):
        #    import pdb
        #    pdb.set_trace()
        # if user_to_output is not None:
        #    h_s = h_s.detach()+user_to_output

        logit = self.final_activation(self.hidden_to_output(h_s))  # (B,C)
        # print(logit.shape)
        # print(hidden.shape)
        # print('wwwww')
        return logit, hidden

    def forward(self, input_x, hidden):
        """
            Faster version of forward torch.jit powered
        """
        #embedded = self.onehot_encode(input_x)

        #h_s = self.input_to_session(embedded)
        h_s = self.embedding_input(input_x)
        Hs_new = []
        for i in range(len(self.session_layers)):
            h_s = self.gru(h_s, hidden[i])
            h_s = self.dropout_hidden(h_s)
            Hs_new.append(h_s)
        hidden = torch.stack(Hs_new)

        logit = self.final_activation(self.hidden_to_output(h_s))

        return logit, hidden

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input

    def init_emb(self):

        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)

        return onehot_buffer

    def onehot_encode(self, input):

        buffer = self.onehot_buffer.view(self.batch_size, self.input_size).zero_()
        index = input.view(-1, 1)
        buffer.scatter_(1, index, 1)
        return buffer

    def init_hidden(self):

        h0 = torch.zeros(len(self.session_layers), self.batch_size, self.hidden_size).to(self.device)
        return h0


class UserGRU(jit.ScriptModule):
    __constants__ = ['user_layers']

    def __init__(self, input_size, hidden_size, output_size,
                 user_layers=[100],
                 dropout_p_hidden_usr=.5,
                 dropout_p_init=.5,
                 user_to_session_act='tanh',
                 dropout_input=0, batch_size=50, use_cuda=True):
        super(UserGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.user_layers = user_layers
        self.hidden_size = hidden_size

        self.dropout_input = dropout_input
        self.dropout_p_hidden_usr = dropout_p_hidden_usr
        self.dropout_p_init = dropout_p_init

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.hidden_activation = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(user_layers[0])
        self.dropout_hidden = nn.Dropout(p=dropout_p_hidden_usr)

        self.user_to_output = MatrixLayer(user_layers[0], output_size)
        self.user_input = MatrixLayer(input_size, user_layers[0])

        self.dropout_output = nn.Dropout(p=self.dropout_p_init)
        if user_to_session_act == 'tanh':
            self.user_to_output_act = nn.Tanh()
        elif user_to_session_act == 'relu':
            self.user_to_output_act = nn.ReLU()
        else:
            raise NotImplementedError('user-to-session activation {} not implemented'.format(user_to_session_act))

        self.gru = ParallelGRU(input_size, hidden_size)

        self = self.to(self.device)

    def forward(self, input, Hu, Sstart, Ustart):
        h_u = self.user_input(input)
        Hu_new = []
        for i in range(len(self.user_layers)):
            h_u = self.gru(h_u, Hu[i])
            h_u = self.dropout_hidden(h_u)
            Hu_new.append(h_u)

        Hu = torch.stack(Hu_new)
        output_user = self.dropout_output(self.user_to_output_act(self.user_to_output(h_u)))
        return output_user, Hu

    def adv_forward(self, input, Hu, Sstart, Ustart, delta_adv={}):
        h_u = self.user_input(input)
        Hu_new = []
        for i in range(len(self.user_layers)):
            h_u = self.gru.adv_forward(h_u, Hu[i], delta_adv)
            h_u = self.dropout_hidden(h_u)
            Hu_new.append(h_u)

        Hu = torch.stack(Hu_new)
        output_user = self.dropout_output(self.user_to_output_act(self.user_to_output(h_u)))
        return output_user, Hu

    def init_hidden(self):

        h_u = torch.zeros(len(self.user_layers), self.batch_size, self.hidden_size).to(self.device)
        return h_u


class HGRU(nn.Module):

    def __init__(self, user_gru, session_gru, discriminator=None, gaussian_mode=2, use_cuda=True):
        super(HGRU, self).__init__()
        self.user_gru = user_gru.cuda() if user_gru is not None else None
        self.session_gru = session_gru.cuda() if session_gru is not None else None
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.discriminator = discriminator if discriminator is not None else None

        from modules.gaussian_noise import GaussianNoise
        self.gaussian = GaussianNoise(sigma=1e-5)
        self.gaussian_mode = gaussian_mode
        self = self.to(self.device)

    def reset_hidden(self, hidden, mask):

        mask = np.arange(len(mask))[mask > 0]
        if len(mask) != 0:
            hidden[:, mask, :] = 0
        return hidden

    def mask_hidden_state(self, h_s, h_u, Sstart, Ustart, mode='default'):
        '''
        This function handles the resetting of states when we encounter new sessions or new users
        :param h_s: session level hidden state
        :param h_u: user level hidden state
        :param Sstart: reset mask for session hidden state
        :param Ustart: reset mask for user hidden state
        :param mode: `default`,`apr`,`discriminator`
        :return: Tensor
        '''
        if mode == 'default':
            # creating indexes for rows to reset
            not_sstart_idx = (1 - Sstart).nonzero().flatten()  # (1- Sstart)
            sstart_idx = Sstart.nonzero().flatten()  # Sstart
            not_ustart_idx = (1 - Ustart).nonzero().flatten()  # (1-Ustart)

            rows = torch.zeros_like(h_s)
            # selecting and copying rows (Hs*(1-Sstart)+h_s_init * Sstart)
            session_rows = h_s.index_select(0, not_sstart_idx)
            user_rows = h_u.index_select(0, sstart_idx)
            rows.index_copy_(0, not_sstart_idx, session_rows)
            rows.index_copy_(0, sstart_idx, user_rows)

            mixed_rows = torch.zeros_like(rows)
            # selecting rows (h_s*(1-Ustart)+ T.zeros_like(h_s)*Ustart)
            remaining_user_rows = rows.index_select(0, not_ustart_idx)
            mixed_rows.index_copy_(0, not_ustart_idx, remaining_user_rows)

            return mixed_rows.unsqueeze(0)

        if mode == 'apr':
            # creating indexes for rows to reset
            not_sstart_idx = (1 - Sstart).nonzero().flatten()  # (1- Sstart)
            sstart_idx = Sstart.nonzero().flatten()  # Sstart
            not_ustart_idx = (1 - Ustart).nonzero().flatten()  # (1-Ustart)

            rows = torch.zeros_like(h_s)
            # selecting and copying rows (Hs*(1-Sstart)+h_s_init * Sstart)
            session_rows = h_s.index_select(0, not_sstart_idx)
            user_session_rows = h_u.index_select(0, not_sstart_idx)
            session_rows = user_session_rows + session_rows
            user_rows = h_u.index_select(0, sstart_idx)

            rows.index_copy_(0, not_sstart_idx, session_rows)
            rows.index_copy_(0, sstart_idx, user_rows)

            mixed_rows = torch.zeros_like(rows)
            # selecting rows (h_s*(1-Ustart)+ T.zeros_like(h_s)*Ustart)
            remaining_user_rows = rows.index_select(0, not_ustart_idx)
            mixed_rows.index_copy_(0, not_ustart_idx, remaining_user_rows)

            return mixed_rows.unsqueeze(0)

        if mode == 'replace':
            # creating indexes for rows to reset
            not_sstart_idx = (1 - Sstart).nonzero().flatten()  # (1- Sstart)
            sstart_idx = Sstart.nonzero().flatten()  # Sstart
            not_ustart_idx = (1 - Ustart).nonzero().flatten()  # (1-Ustart)

            rows = torch.zeros_like(h_s)
            # selecting and copying rows (Hs*(1-Sstart)+h_s_init * Sstart)
            session_rows = h_s.index_select(0, not_sstart_idx)
            user_session_rows = h_u.index_select(0, not_sstart_idx)
            session_rows = session_rows + user_session_rows
            user_rows = h_u.index_select(0, sstart_idx)

            rows.index_copy_(0, not_sstart_idx, session_rows)
            rows.index_copy_(0, sstart_idx, user_rows)

            mixed_rows = torch.zeros_like(rows)
            # selecting rows (h_s*(1-Ustart)+ T.zeros_like(h_s)*Ustart)
            remaining_user_rows = rows.index_select(0, not_ustart_idx)
            mixed_rows.index_copy_(0, not_ustart_idx, remaining_user_rows)

            return mixed_rows.unsqueeze(0)

        if mode == 'discriminator':
            return

    def forward(self, input, Sstart, Ustart, Hs, Hu):

        h_s_init, new_Hu = self.user_gru(Hs[-1], Hu, Sstart, Ustart)

        h_s = self.mask_hidden_state(Hs[-1], h_s_init, Sstart, Ustart)

        logit, new_Hs = self.session_gru(input, h_s)
        return logit, new_Hs, new_Hu

    def forward_init(self, input, Sstart, Ustart, Hs, Hu):

        h_s_init, new_Hu = self.user_gru(Hs[-1], Hu, Sstart, Ustart)

        h_s = self.mask_hidden_state(Hs[-1], h_s_init, Sstart, Ustart)

        logit, new_Hs = self.session_gru(input, h_s)
        return logit, new_Hs, new_Hu, h_s_init

    def forward_discriminator(self, input, Sstart, Ustart, Hs, Hu, H_s_init, mask_discriminate, session_steps):
        # Sstart, Ustart, Hs, Hu, return_x_state=True,strategy='replace', steps=None
        h_s_init, new_Hu = self.user_gru(Hs[-1], Hu, Sstart, Ustart)
        state = self.get_state(input, Sstart, Ustart, Hs, Hu, H_s_init, False, self.discriminator.strategy,
                               session_steps)
        #print(state[2])
        action = self.discriminator(state)
        # action = action * mask_discriminate
        strategy = self.discriminator.strategy
        #print(action.mean(1)*mask_discriminate)
        Hs, Sstart = self.handle_action(Hs, h_s_init, action, strategy, Sstart, mask_discriminate)

        # Sstart += action
        # h_s_init = action * h_s_init
        h_s = self.mask_hidden_state(Hs[-1], h_s_init, Sstart, Ustart)
        logit, Hs = self.session_gru(input, h_s)

        return logit, Hs, new_Hu

    def get_state(self, input, Sstart, Ustart, Hs, Hu, H_s_init, return_x_state=True,
                  strategy='replace', steps=None):
        state_components = []
        h_s_init, Hu = self.user_gru(Hs[-1], Hu, Sstart, Ustart)
        h_u_state = h_s_init
        h_s_state = Hs[-1]
        h_u_init = H_s_init  # Sstart * h_s_state
        if return_x_state:
            embedded = self.session_gru.onehot_encode(input)
            x_state = self.session_gru.input_to_session(embedded)
            state_components = [h_s_state, h_u_state, x_state]
        else:
            state_components = [h_s_state, h_u_state]

        if strategy == 'add-noise-init':
            steps = torch.log(steps)
            state_components = [h_u_init, steps, h_u_state, steps, h_s_state, steps]
        if strategy == 'gate-init':
            steps = torch.log(steps)
            state_components = [h_u_init, steps, h_u_state, steps, h_s_state, steps]
        if strategy == 'replace':
            steps = torch.log(steps)
            state_components = [h_u_init, steps, h_u_state, steps, h_s_state, steps]
        if strategy == 'gate-init2':
            steps = torch.log(steps)
            state_components = [h_u_init, steps, h_u_state, steps, h_s_state, steps]
        # print([str(x.size()) for x in state_components])
        return torch.cat(state_components, 1).to(self.device)

    def step(self, input, Sstart, Ustart, Hs, Hu, action, mask_discriminate, strategy='replace'):

        Hs, Sstart = self.handle_action(Hs, Hu, action, strategy, Sstart, mask_discriminate)
        #Hs_clamp = [h_s.clamp(-1.0, 1.0) for h_s in Hs]
        logit, Hs, Hu, H_s_init = self.forward_init(input, Sstart, Ustart, Hs, Hu)
        return logit, Hs, Hu, H_s_init

    def handle_action(self, Hs, Hu, action, strategy, Sstart, mask_discriminate):
        if strategy == 'replace':
            # Complete restart of  session init state (batch_size, 1)
            action = action * mask_discriminate
            Sstart = ((Sstart + action) > 0.0).float()
        elif strategy == 'add-noise':
            # Adding transformation of states Hu Hs to Hs (batch_size, hidden_size)
            action = action * mask_discriminate
            Hs = Hs + action
        elif strategy == 'add-noise-init':
            # Adding transformation of states Hu Hs to Hs (batch_size, hidden_size)
            action = action * mask_discriminate
            Hs = Hs + action
            Hs = Hs.clamp(-1.0, 1.0)
        elif strategy == 'gate-init':
            # Adding transformation of states Hu Hs to Hs (batch_size, hidden_size)
            Hs = (Hs * (1.-action) + action * Hu) * mask_discriminate   + Hs * (1 - mask_discriminate)
            # Hs = Hs * mask_discriminate * action + Hs * (1 - mask_discriminate)
        elif strategy == 'gate-init2':
            # Adding transformation of states Hu Hs to Hs (batch_size, hidden_size)
            action = action * mask_discriminate
            Hs = Hs * action + Hs * (1 - mask_discriminate)
        elif strategy == 'gate':
            # Multiply by mask (batch_size, hidden_size)
            Hs = Hs * mask_discriminate * action + (1.0 - action) * Hu + Hs(1 - mask_discriminate)
        elif strategy == 'binary-mask':
            # Multiply by mask (batch_size, hidden_size)
            Hs = Hs + action * Hu
        else:
            raise NotImplementedError('Not implemented strategy mode')

        return Hs, Sstart

    def merge_action_init(self, Hs_0, h_s_init, Sstart, action):
        if action is not None:
            if self.mask_mode == 'batch_wise':
                new_Hs_0 = (Hs_0 + action + h_s_init) * (1.0 - Sstart) + h_s_init * Sstart
            else:
                # new_Hs_0 = ((1.0 - action) * Hs_0 + action * h_s_init) * (1.0 - Sstart) + h_s_init * Sstart
                new_Hs_0 = (Hs_0 + action) * (1.0 - Sstart) + h_s_init * Sstart
        else:
            new_Hs_0 = Hs_0
        return new_Hs_0

    def get_reset_state(self, batch_size=100, state_size=300):

        return torch.zeros(batch_size, state_size).to(self.device)

    def reset(self, path):
        return self.load(path)

    def load(self, path):
        return torch.load(path)

    def adv_forward(self, input, Sstart, Ustart, Hs, Hu, noise):

        h_s_init, Hu = self.user_gru.adv_forward(Hs[-1], Hu, Sstart, Ustart, noise)
        h_s = self.mask_hidden_state(Hs[-1], h_s_init, Sstart, Ustart, mode='apr')
        logit, Hs = self.session_gru(input, h_s)

        return logit, Hs, Hu

    def create_adversarial(self, mode='grad', eps=0.5, mean=0.0, std=0.1):
        """
        This function should be called after the backward call
        :param mode: 'grad' or 'random'
        :param eps:
        :param mean:
        :param std:
        :return: noise dictionary of adversarial noise for weights
        """

        noise = {}

        if mode == 'random':
            # Sampling from normal distribution
            noise['Wxh'] = torch.zeros_like(self.user_gru.gru.Wxh).normal_(mean, std) * eps
            noise['Whh'] = torch.zeros_like(self.user_gru.gru.Whh).normal_(mean, std) * eps
            noise['bxh'] = torch.zeros_like(self.user_gru.gru.bxh).normal_(mean, std) * eps
            noise['bhh'] = torch.zeros_like(self.user_gru.gru.bhh).normal_(mean, std) * eps
        if mode == 'grad':
            params_to_check = ['Wxh', 'Whh', 'bxh', 'bhh']
            i = 0
            for name, param in self.user_gru.gru.named_parameters():
                if name in [params_to_check[i]]:
                    key = '' + name
                    if param.grad is not None:
                        noise[key] = param.grad.norm(dim=0).detach().to(self.device) * eps
                    else:
                        noise[key] = torch.zeros_like(param)

                    i += 1
                if i > len(params_to_check) - 1:
                    break
        if mode == 'zeros':
            noise = {'Wxh': 0, 'Whh': 0, 'bxh': 0, 'bhh': 0}

        return noise


class UserVAE(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 user_layers=[100],
                 dropout_p_hidden_usr=.5,
                 dropout_p_init=.5,
                 user_to_session_act='tanh',
                 latent_size=50,
                 dropout_input=0, batch_size=50, use_cuda=True):
        super(UserGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.user_layers = user_layers
        self.hidden_size = hidden_size

        self.latent_size = latent_size

        self.dropout_input = dropout_input
        self.dropout_p_hidden_usr = dropout_p_hidden_usr
        self.dropout_p_init = dropout_p_init

        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.hidden_activation = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(user_layers[0])
        self.dropout_hidden = nn.Dropout(p=dropout_p_hidden_usr)
        # self.user_gru = nn.GRU(input_size=input_size, hidden_size=user_layers[0], num_layers=user_layers[0],
        #                       dropout=self.dropout_p_hidden_usr)
        # self.user_gru = MultilayerRNN(batch_size=batch_size,
        #                         input_size=input_size,
        #                         hidden_size=hidden_size,
        #                         num_layers=user_layers,
        #                         dropout_hidden=self.dropout_p_hidden_usr)

        self.user_to_output = MatrixLayer(user_layers[0], output_size)
        self.user_input = MatrixLayer(input_size, user_layers[0])

        self.dropout_output = nn.Dropout(p=self.dropout_p_init)
        if user_to_session_act == 'tanh':
            self.user_to_output_act = nn.Tanh()
        elif user_to_session_act == 'relu':
            self.user_to_output_act = nn.ReLU()
        else:
            raise NotImplementedError('user-to-session activation {} not implemented'.format(user_to_session_act))
        # self.handler = PropagationHandler(input_size*2, output_size)

        self.Wu_in = nn.Parameter(torch.Tensor(1, user_layers[0], 3 * user_layers[0]))
        self.Bu_h = nn.Parameter(torch.Tensor(1, 3 * user_layers[0]))
        self.Wu_hh = nn.Parameter(torch.Tensor(1, user_layers[0], user_layers[0]))
        self.Wu_rz = nn.Parameter(torch.Tensor(1, user_layers[0], 2 * user_layers[0]))
        self.init_weights()
        self = self.to(self.device)

    def init_weights(self):
        # nn.init.uniform_(self.Wu_in)
        # nn.init.uniform_(self.Bu_h)
        # nn.init.uniform_(self.Wu_hh)
        # nn.init.uniform_(self.Wu_rz)
        nn.init.orthogonal_(self.Wu_in)
        nn.init.orthogonal_(self.Bu_h)
        nn.init.orthogonal_(self.Wu_hh)
        nn.init.orthogonal_(self.Wu_rz)

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.latent_size]
        log_sigma = temp_out[:, self.latent_size:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, input, Hu, Sstart, Ustart):
        # encoder

        input = self.batchnorm(input)
        h_u = self.user_input(input)  # .unsqueeze(0)
        # output_user, Hu = self.user_gru(user_in, Hu)
        Hu_new = []
        for i in range(len(self.user_layers)):
            user_in = torch.mm(h_u, self.Wu_in[i]) + self.Bu_h
            user_in = user_in.t()

            rz_u = torch.sigmoid(user_in[self.user_layers[0]:] + (torch.mm(Hu[i], self.Wu_rz[i])).t())

            h_u = self.hidden_activation(torch.mm(Hu[i] * rz_u[:self.user_layers[0]].t(), self.Wu_hh[i]).t()
                                         + user_in[:self.user_layers[0]])
            z = rz_u[self.user_layers[0]:].t()
            h_u = (1.0 - z) * Hu[i] + z * h_u.t()
            h_u = self.dropout_hidden(h_u)
            h_u = Hu[i] * (1.0 - Sstart) + h_u * Sstart
            h_u = h_u * (1.0 - Ustart)
            Hu_new.append(h_u)

        Hu = torch.stack(Hu_new)

        h_u = self.sample_latent(h_u)
        # decoder
        output_user = self.dropout_output(self.user_to_output_act(self.user_to_output(h_u)))
        return output_user, Hu

    def init_hidden(self):

        h_u = torch.zeros(len(self.user_layers), self.batch_size, self.hidden_size).to(self.device)
        return h_u
