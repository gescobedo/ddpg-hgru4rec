import torch
from torch import nn
from torch.distributions import Categorical, Bernoulli
import torch.nn.functional as F
from modules.layer import MatrixLayer
from modules.gaussian_noise import GaussianNoise

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs=300, num_actions=100, hidden_size=256, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        # self.linear1 = MatrixLayer(num_inputs + num_actions, hidden_size)
        # self.linear2 = MatrixLayer(hidden_size, hidden_size)
        # self.linear3 = MatrixLayer(hidden_size, 1)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # self.linear4 = MatrixLayer(2*hidden_size,   hidden_size)
        self.act_funtion = nn.Sigmoid()##nn.ReLU()
        self.bn_state = nn.BatchNorm1d(num_inputs)
        self.bn_action = nn.BatchNorm1d(num_actions)
        self.bn_hidden = nn.BatchNorm1d(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)
        self = self.to(device)

    def forward(self, state, action):
        state = self.bn_state(state)
        action = self.bn_action(action)
        x = torch.cat([state, action], 1)
        # x =self.bn(x)
        x = self.act_funtion(self.linear1(x))

        # x = self.bn_hidden(x)
        x = self.act_funtion(self.linear2(x))
        # x = self.act_funtion(self.linear4(x))
        x = self.act_funtion(self.linear3(x))
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs=300, num_actions=100, hidden_size=256, strategy='replace', init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.strategy = strategy
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(100)
        self.bn_hidden = nn.BatchNorm1d(hidden_size)
        self.gru_cell = nn.GRUCell(num_actions, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.create_layers(self.num_inputs, self.num_actions, self.hidden_size)
        self.config = 0
        self = self.to(device)

    def create_layers(self, num_inputs, num_actions, hidden_size):

        if self.strategy == 'replace':
            i_size = int(num_inputs / 3)
            self.linear1 = nn.Linear(i_size, hidden_size)
            self.linear2 = nn.Linear(i_size, hidden_size)
            self.linear3 = nn.Linear(i_size, hidden_size)
            self.linear4 = nn.Linear(hidden_size , num_actions)

        elif self.strategy == 'choice':
            i_size = int(num_inputs / 2)
            self.linear1 = nn.Linear(i_size, hidden_size)
            self.linear2 = nn.Linear(i_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, hidden_size)
            self.linear4 = nn.Linear(hidden_size, num_actions)

        elif self.strategy == 'add-noise':
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, num_actions)

        elif self.strategy == 'add-noise-init':
            i_size = int(num_inputs / 3)
            self.linear1 = nn.Linear(i_size, hidden_size)
            self.linear2 = nn.Linear(i_size, hidden_size)
            self.linear3 = nn.Linear(i_size, hidden_size)
            self.linear4 = nn.Linear(hidden_size * 2, num_actions)
            self.e1 = nn.Linear(i_size*2,i_size)
            self.e2 =nn.Linear(i_size*2,i_size)
            # self.linear1 = nn.Linear(num_inputs, 3 * hidden_size)
            # self.linear2 = nn.Linear(hidden_size, hidden_size)
            # self.linear3 = nn.Linear(hidden_size, num_actions)
            # self.linear4 = nn.Linear(hidden_size, num_actions)
        elif self.strategy == 'gate-init':
            i_size = int(num_inputs / 3)
            self.linear1 = nn.Linear(i_size, hidden_size)
            self.linear2 = nn.Linear(i_size, hidden_size)
            self.linear3 = nn.Linear(i_size, hidden_size)
            self.linear4 = nn.Linear(hidden_size*2, num_actions)
        elif self.strategy == 'gate-init2':
            i_size = int(num_inputs / 3)
            self.linear1 = nn.Linear(2 * i_size, hidden_size)
            self.linear2 = nn.Linear(i_size, hidden_size)
            self.linear3 = nn.Linear(2 * i_size, hidden_size)
            self.linear4 = nn.Linear(2 * hidden_size, num_actions)
        elif self.strategy == 'gate':
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, num_actions)

        elif self.strategy == 'binary-mask':
            self.linear1 = nn.Linear(num_inputs, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, num_actions)
        else:
            raise NotImplementedError('Not implemented strategy mode')

    def forward(self, state):

        original_user = self.bn(state[:, :100])
        original_sess = self.bn(state[:, 100:200])
        if self.strategy == 'replace':
            i_size = int(self.num_inputs / 3)

            i_h_u_init, i_h_u_cand, i_h_s = state[:, :i_size], \
                                            state[:, i_size:2 * i_size], \
                                            state[:, -i_size:]
            # e_h_u_init = self.relu(self.e1(torch.cat([i_h_u_init,i_h_u_cand], 1)))
            # e_h_s = self.relu(self.e2(torch.cat([i_h_s,i_h_u_cand], 1)))
            h_u_init = self.sigmoid(self.linear1(i_h_u_init))
            h_u_cand = self.tanh(self.linear2(i_h_u_cand))

            # h_s = self.tanh(x[:, -self.hidden_size:])
            h_s_gate = self.sigmoid(self.linear3(i_h_s))
            # print([h_u_cand.shape,h_u_init.shape, h_s.shape])
            out = h_u_init * h_u_cand - h_s_gate * h_u_cand
            out = self.tanh(self.linear4(out))
            return out

        elif self.strategy == 'choice':
            x = self.relu(self.linear1(torch.cat([original_user, original_sess], 1)))
            x = self.relu(self.linear2(x))
            x = self.tanh(self.linear3(x))
            return x
        elif self.strategy == 'add-noise':
            x = self.relu(self.linear1(torch.cat([original_user, original_sess], 1)))
            x = self.relu(self.linear2(x))
            x = self.tanh(self.linear3(x))
            return x
        elif self.strategy == 'add-noise-init':
            # # 3*h_size ->
            # x = self.linear1(state)
            # h_u_init = self.sigmoid(x[:, :self.hidden_size])
            # h_u_cand = self.relu(x[:, self.hidden_size:2 * self.hidden_size])
            #
            # h_s = self.tanh(x[:, -self.hidden_size:])
            # h_s_gate = self.sigmoid(x[:, -self.hidden_size:])
            # # print([h_u_cand.shape,h_u_init.shape, h_s.shape])
            # out = h_u_init * h_u_cand + h_s_gate * h_u_cand
            # out = self.tanh(self.linear3(out))
            # return out

            i_size = int(self.num_inputs / 3)


            i_h_u_init, i_h_u_cand, i_h_s = state[:, :i_size], \
                                            state[:, i_size:2 * i_size], \
                                            state[:, -i_size:]
            #e_h_u_init = self.relu(self.e1(torch.cat([i_h_u_init,i_h_u_cand], 1)))
            #e_h_s = self.relu(self.e2(torch.cat([i_h_s,i_h_u_cand], 1)))
            if self.config == 3:
                step_mask = torch.ones(i_h_u_cand.shape).to(device)
                step_mask[:-1] =0.0
                i_h_u_cand=i_h_u_cand*step_mask+1.0*(1.0-step_mask)
                i_h_u_init = i_h_u_init * step_mask + 1.0 * (1.0 - step_mask)
                i_h_s = i_h_s * step_mask + 1.0 * (1.0 - step_mask)

            h_u_init = self.sigmoid(self.linear1(i_h_u_init))
            h_u_cand = self.tanh(self.linear2(i_h_u_cand))

            # h_s = self.tanh(x[:, -self.hidden_size:])
            h_s_gate = self.sigmoid(self.linear3(i_h_s))
            # print([h_u_cand.shape,h_u_init.shape, h_s.shape])
            if self.config == 1:
                h_u_init = 1.0
            if self.config == 2:
                h_s_gate =1.0



            out = torch.cat([h_u_init * h_u_cand, h_s_gate * h_u_cand], 1)
            out = self.tanh(self.linear4(out))
            return out

        elif self.strategy == 'gate-init':
            # 3*h_size ->

            i_size = int(self.num_inputs / 3)

            i_h_u_init, i_h_u_cand, i_h_s = state[:, :i_size], \
                                            state[:, i_size:2 * i_size], \
                                            state[:, -i_size:]
            if self.config == 3:
                step_mask = torch.ones(i_h_u_cand.shape).to(device)
                step_mask[:-1] = 0.0
                i_h_u_cand = i_h_u_cand * step_mask + 1.0 * (1.0 - step_mask)
                i_h_u_init = i_h_u_init * step_mask + 1.0 * (1.0 - step_mask)
                i_h_s = i_h_s * step_mask + 1.0 * (1.0 - step_mask)

            h_u_init = self.sigmoid(self.linear1(i_h_u_init))
            h_u_cand = self.tanh(self.linear2(i_h_u_cand))

            # h_s = self.tanh(x[:, -self.hidden_size:])
            h_s_gate = self.sigmoid(self.linear3(i_h_s))
            # print([h_u_cand.shape,h_u_init.shape, h_s.shape])
            if self.config == 1:
                h_u_init = 1.0
            if self.config == 2:
                h_s_gate= 1.0

            out = torch.cat([h_u_init * h_u_cand , h_s_gate * h_u_cand], 1)
            out = self.sigmoid(self.linear4(out))
            return out
        elif self.strategy == 'gate-init2':
            # 3*h_size ->

            i_size = int(self.num_inputs / 3)

            i_h_u_init, i_h_u_cand, i_h_s = state[:, :2 * i_size], state[:, i_size:2 * i_size], state[:, i_size:]
            h_u_init = self.sigmoid(self.linear1(i_h_u_init))
            h_u_cand = self.relu(self.linear2(i_h_u_cand))
            h_s_gate = self.sigmoid(self.linear3(i_h_s))
            out = torch.cat([h_u_init * h_u_cand, h_s_gate * h_u_cand], 1)
            out = self.sigmoid(self.linear4(out))
            return out
        elif self.strategy == 'gate':
            x = self.relu(self.linear1(torch.cat([original_user, original_sess], 1)))
            x = self.relu(self.linear2(x))
            x = self.sigmoid(self.linear3(x))
            return x

        elif self.strategy == 'binary-mask':
            x = self.relu(self.linear1(torch.cat([original_user, original_sess], 1)))
            x = self.relu(self.linear2(x))
            x = self.tanh(self.linear3(x))
            x = Bernoulli(logits=x)
            return x.sample()
        else:
            raise NotImplementedError('Not implemented strategy mode')

    def get_action(self, state):
        # state = torch.Tensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.float()


class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    # state_dim, action_dim, hidden_dim
    def __init__(self, num_classes=100, embedding_dim=100, filter_sizes=[1, 2], num_filters=[50, 60], dropout_prob=0.2):
        super(Discriminator, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(sum(num_filters), num_classes)
        self.embedding_dim = embedding_dim
        self.batch_norm = nn.BatchNorm2d(1)
        self = self.to(device)

    def forward(self, x):
        """
        Inputs: x
            - x: (batch_size, seq_len)
        Outputs: out
            - out: (batch_size, num_classes)
        """
        x = x.view(-1, 1, 2, self.embedding_dim)
        x = self.batch_norm(x)
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * seq_len]
        # print(convs[0].shape)
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        highway = self.highway(out)
        transform = F.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out  # sets C = 1 - T
        out = F.log_softmax(self.fc(self.dropout(out)), dim=1)  # batch * num_classes
        # print(out.shape)
        return out

    def get_action(self, state):
        # state = torch.Tensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach()


class CNNPolicy(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.linear1 = nn.Linear(32 * 7 * 7, 512)
        self.critic_linear = nn.Linear(512, action_space)
        # self.dist = Categorical(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        x = F.relu(self.conv1(inputs / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x)

    def get_action(self, state):
        # state = torch.Tensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach()


class VariationalPolicy(nn.Module):
    def __init__(self, hidden_size, latent_size, hidden_drop=0.2, out_drop=0.1):
        super(VariationalPolicy, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        # self.fc0 = MatrixLayer(3 * hidden_size, 2 * latent_size)
        self.fc1 = MatrixLayer(1 * hidden_size, hidden_size)
        self.fc2 = MatrixLayer(2 * hidden_size, hidden_size)
        self.fc3 = MatrixLayer(hidden_size, 2 * latent_size)
        self.fc4 = MatrixLayer(latent_size, hidden_size)

        self.act_fun = nn.ReLU()
        self.out_act_fun = nn.Tanh()

        self.hidden_dropout = nn.Dropout(hidden_drop)
        self.out_dropout = nn.Dropout(out_drop)

        self.batchnorm_input = nn.BatchNorm1d(hidden_size)

        self.log_softmax = nn.LogSoftmax()

    def sample_latent(self, h_enc):
        """
               Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = h_enc

        mu = temp_out[:, :self.latent_size]
        log_sigma = temp_out[:, self.latent_size:]

        sigma = torch.exp(log_sigma)
        std_z = torch.zeros(self.latent_size).normal_().to(device)

        self.z_mean = mu
        self.z_log_sigma = log_sigma
        # Reparameterization trick
        return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False)

    def forward(self, state):
        h_s, h_u, x_embed = state[:, :self.hidden_size], \
                            state[:, self.hidden_size:2 * self.hidden_size], \
                            state[:, 2 * self.hidden_size:]
        # Encoder
        # h_u_x =self.hidden_dropout(self.fc1(torch.cat([h_u, x_embed], -1)))

        # h_s_x = self.hidden_dropout(self.fc2(torch.cat([h_s, x_embed], -1)))
        # h_s_x = self.fc2(torch.cat([h_s, x_embed], -1))
        h_s_x = self.log_softmax(self.fc2(torch.cat([h_s, h_u], -1)))

        encoder_out = self.fc3(h_s_x * h_s + (1.0 - h_s_x) * self.fc1(h_u))

        # Sampling
        sampled_z = self.sample_latent(encoder_out)

        # Decoder
        decoder_out = self.out_act_fun(self.fc4(sampled_z))

        return decoder_out, self.z_mean, self.z_log_sigma

    def get_action(self, state):
        # state = torch.Tensor(state).unsqueeze(0).to(device)
        action, _, _ = self.forward(state)
        return action.detach()
