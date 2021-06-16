import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import device
from torch import optim
from modules.optimizer import Optimizer
from modules.loss import LossFunction
from modules.layer import SessionGRU, HGRU, UserGRU
import modules.evaluate as E
from modules.data import SessionDataset, SessionDataLoader
from tqdm import tqdm
from datetime import datetime


class HGRU4REC(object):


    def __init__(self, input_size, hidden_size, output_size, session_layers, user_layers, n_epochs=10, batch_size=50,
                 learning_rate=.05, momentum=0, adapt='Adagrad', decay=0.9, grad_cap=-1, sigma=0,
                 dropout_p_hidden_usr=0.0, dropout_p_hidden_ses=0.0, dropout_p_init=0.0, init_as_normal=False,
                 reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None, train_random_order=False,
                 lmbd=0.0, session_key='SessionId', item_key='ItemId', time_key='Time', user_key='UserId', n_sample=0,
                 eps=1e-6, sample_alpha=0.75, item_embedding=None, init_item_embeddings=None,
                 user_propagation_mode='init', user_to_output=False, user_to_session_act='tanh',
                 dropout_input=.0, seed=42, use_cuda=True, time_sort=False, pretrained=None):

        """ The HGRU4REC model

        """

        self.user_layers = user_layers
        self.session_layers = session_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden_usr = dropout_p_hidden_usr
        self.dropout_p_hidden_ses = dropout_p_hidden_ses
        self.dropout_p_init = dropout_p_init
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.reset_after_session = reset_after_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.user_key = user_key
        self.train_random_order = train_random_order
        # Negative Sampling
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha

        self.user_propagation_mode = user_propagation_mode
        self.user_to_output = user_to_output

        self.item_embedding = item_embedding
        self.init_item_embeddings = init_item_embeddings

        self.lmbd = lmbd
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if pretrained is None:
            self.hgru = self.create_model()
        else:
            self.hgru = pretrained

        # Initialize the optimizer
        self.adapt = adapt
        self.weight_decay = decay
        self.momentum = momentum
        self.lr = learning_rate
        self.eps = eps

        # DQN for improving personaliztion during session
        # This should be a pretrained instance

        self.optimizer_session = Optimizer(params=self.hgru.parameters(),
                                           optimizer_type=adapt,
                                           lr=learning_rate,
                                           weight_decay=decay,
                                           momentum=momentum,
                                           eps=eps)

        # Initialize the loss function
        self.loss_type = loss
        self.loss_fn = LossFunction(loss, use_cuda)

        # gradient clipping(optional)
        self.clip_grad = grad_cap

        # etc
        self.time_sort = time_sort
        self.train_error = False

    def create_model(self):
        session_gru = SessionGRU(self.input_size,
                                 self.hidden_size,
                                 self.output_size,
                                 session_layers=self.session_layers,
                                 dropout_hidden=self.dropout_p_hidden_ses,
                                 dropout_input=0,
                                 batch_size=self.batch_size,
                                 use_cuda=self.use_cuda
                                 )

        user_gru = UserGRU(self.hidden_size,
                           self.hidden_size,
                           self.hidden_size,
                           user_layers=self.user_layers,
                           dropout_p_hidden_usr=self.dropout_p_hidden_usr,
                           dropout_p_init=self.dropout_p_init,
                           dropout_input=0,
                           batch_size=self.batch_size,
                           use_cuda=self.use_cuda
                           )

        return HGRU(user_gru, session_gru, use_cuda=self.use_cuda)


    def run_epoch(self, dataset, k=20, training=True, batch_wise_eval=False):
        """ Run a single training epoch """
        start_time = datetime.now()

        # initialize
        losses = []
        recalls = []
        mrrs = []
        eval_data = pd.DataFrame([])
        eval_cutoffs = [5, 10, 20]
        optimizer_session = self.optimizer_session
        # optimizer_user = self.optimizer_user
        Hs, Hu = self.hgru.session_gru.init_hidden(), self.hgru.user_gru.init_hidden()
        if not training:
            self.hgru.eval()

        device = self.device

        # Start the training loop
        loader = SessionDataLoader(dataset, batch_size=self.batch_size)
        count = 0
        for input, target, Sstart, Ustart, in_eval in loader:
            count += Sstart.sum()
            input = input.to(device)
            target = target.to(device)
            Sstart = torch.Tensor(Sstart[:, None]).to(device)
            Ustart = torch.Tensor(Ustart[:, None]).to(device)
            logit, Hu, Hs = self.hgru(input, Sstart, Ustart, Hs.detach(), Hu.detach())

            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_fn(logit_sampled)
            losses.append(loss.item())

            eval_data = self.evaluate(batch_wise_eval, eval_cutoffs, eval_data, in_eval, loss, logit, mrrs, recalls,
                                      target,
                                      training)

            if training:
                loss.backward()

                if self.clip_grad != -1:
                    _ = torch.nn.utils.clip_grad_value_(self.hgru.parameters(), self.clip_grad)
                    for p in self.hgru.parameters():
                        import pdb
                        # pdb.set_trace()
                        if p.grad is not None:
                            p.data.add_(self.learning_rate * p.grad)
                optimizer_session.step()
                # self.print_parameters()
                optimizer_session.zero_grad()

                # optimizer_user.step()
                # flush the gradient after the optimization
                # optimizer_user.zero_grad()  # flush the gradient after the optimization

        eval_data, results = self.build_results(batch_wise_eval, eval_cutoffs, eval_data, losses, mrrs, recalls,
                                                start_time)
        print(count)
        return results, eval_data

    def evaluate(self, batch_wise_eval, eval_cutoffs, eval_data, in_eval, loss, logit, mrrs, recalls, target, training):
        if not training:
            with torch.no_grad():
                # if training:
                #    recall, mrr = E.evaluate(logit, target, k)
                # else:
                if logit[in_eval['in_eval']].shape[0] > 0:
                    recall, mrr, ranks = E.evaluate_multiple_with_ranks(logit[in_eval['in_eval']],
                                                                        target[in_eval['in_eval']],
                                                                        eval_cutoffs, batch_wise_eval)

                    if batch_wise_eval:
                        recall = torch.cat(recall, -1)
                        mrr = torch.cat(mrr, -1)
                        in_eval['ranks'] = ranks.cpu().numpy()
                        data_df = pd.DataFrame.from_dict(in_eval)
                        data_df['loss'] = loss.item()
                        eval_data = eval_data.append(data_df, ignore_index=True)
                        # print(eval_data)
                    recalls.append(recall)
                    mrrs.append(mrr)
        else:

            recalls.append([0] * len(eval_cutoffs))
            mrrs.append([0] * len(eval_cutoffs))

        return eval_data

    def build_results(self, batch_wise_eval, eval_cutoffs, eval_data, losses, mrrs, recalls, start_time):
        results = dict()
        results['Loss'] = '{}'.format(np.mean(losses))
        if batch_wise_eval:
            recalls = torch.cat(recalls, 0).cpu().numpy()
            mrrs = torch.cat(mrrs, 0).cpu().numpy()
            print(pd.DataFrame(recalls).describe())
        for idx, k in enumerate(eval_cutoffs):
            if len(recalls) > 0:
                results['R@' + str(k)] = '{}'.format(np.mean(recalls, axis=0)[idx])
                results['MRR@' + str(k)] = '{}'.format(np.mean(mrrs, axis=0)[idx])
            else:
                results['R@' + str(k)] = 'no-value'
                results['MRR@' + str(k)] = 'no-value'
        if batch_wise_eval:
            df_recalls = pd.DataFrame(recalls, columns=['R@' + str(k) for k in eval_cutoffs])
            df_mrrs = pd.DataFrame(mrrs, columns=['MRR@' + str(k) for k in eval_cutoffs])
            eval_data = pd.concat([eval_data, df_recalls, df_mrrs], axis=1)
        end_time = datetime.now()
        results['time'] = str(end_time - start_time)[:-7]
        return eval_data, results

    def train(self, dataset, dataset_test=None, k=20, n_epochs=10, save_dir='../models', save=True,
              model_name='HGRU4REC', resume=False):
        """
        Train the GRU4REC model on a pandas dataframe for several training epochs,
        and store the intermediate models to the user-specified directory.

        Args:
            n_epochs (int): the number of training epochs to run
            save_dir (str): the path to save the intermediate trained models
            model_name (str): name of the model
        """
        tab = " "
        print(f'Training {model_name}...')
        for epoch in range(n_epochs):

            model_fname = f'{model_name}_{self.loss_type}_{self.adapt}_{self.lr}_epoch{epoch + 1:d}'
            save_dir = Path(save_dir)
            if resume:
                resume_file = save_dir / model_fname
                if resume_file.is_file():
                    print(f'Resuming {model_fname}...')
                    self.load_checkpoint(resume_file)
                    self.hgru.train()
                    continue
            if self.train_error:
                break
            results, eval_data = self.run_epoch(dataset, k=k, training=True)
            out_results = [f'{k}:{v}' for k, v in results.items()]
            # self.print_parameters()
            print(f'Epoch:{epoch + 1:2d}{tab}{tab.join(out_results)}')
            if save:
                if not save_dir.exists(): save_dir.mkdir()
                torch.save(self.hgru.state_dict(), save_dir / model_fname)

            if dataset_test is not None:
                self.test(dataset_test, k)
            # Store the intermediate model

    def train_adv(self, dataset, dataset_test=None, k=20, n_epochs=10, save_dir='../models', save=True,
                  model_name='HGRU4REC_ADV',
                  eps=.5, noise_mode='grad', std=.1, mean=.0, lambda_reg=0.5):
        """

        @param dataset:
        @param dataset_test:
        @param k:
        @param n_epochs:
        @param save_dir:
        @param save:
        @param model_name:
        @param eps:
        @param noise_mode:
        @param std:
        @param mean:
        @param lambda_reg:
        """
        tab = " "

        print(f'Creating static weights {model_name}...')
        self.static_model = self.create_model()
        self.static_model.load_state_dict(self.hgru.state_dict())
        self.static_model.eval()

        print(f'Training {model_name}...')
        self.hgru.session_gru.eval()
        self.hgru.user_gru.eval()
        for param in self.hgru.session_gru.parameters():
            param.requires_grad = False
        for name, param in self.hgru.user_gru.named_parameters():
            if name in ['user_to_output.matrix_layer',
                        'user_input.matrix_layer',
                        'user_to_output.bias_layer',
                        'user_input.bias_layer'
                        ]:
                param.requires_grad = False

        for epoch in range(n_epochs):
            if self.train_error:
                break

            results = self.run_adv_epoch(dataset, k=k, training=True,
                                         eps=eps,
                                         noise_mode=noise_mode,
                                         std=std,
                                         mean=mean,
                                         lambda_reg=lambda_reg,
                                         )

            # self.print_parameters()
            # for name, param in self.static_model.named_parameters():
            # print(name)
            #    print("name:{:30.30} norm: {:5f}".format(name, torch.norm(param).item()))
            # print('max_abs :{}'.format(param.abs().max(-1)))
            results = [f'{k}:{v}' for k, v in results.items()]

            print(f'Epoch:{epoch + 1:2d}{tab}{tab.join(results)}')
            if save:
                save_dir = Path(save_dir)
                if not save_dir.exists(): save_dir.mkdir()
                model_fname = f'{model_name}_{self.loss_type}_{self.adapt}_{self.lr}_epoch{epoch + 1:d}'
                torch.save(self.hgru.state_dict(), save_dir / model_fname)

            # if dataset_test is not None:
            #    self.test(dataset_test, k)
            # Store the intermediate model

    def load_checkpoint(self, file_path):
        self.hgru.load_state_dict(torch.load(file_path))
        self.hgru.eval()

    def test(self, dataset, k=20, mode='default', min_steps=10, min_sessions=3, batch_wise_eval=False):
        """ Model evaluation

        Args:
            k (int): the length of the recommendation list

        Returns:
            avg_loss: mean of the losses over the session-parallel minibatches
            avg_recall: mean of the Recall@K over the session-parallel mini-batches
            avg_mrr: mean of the MRR@K over the session-parallel mini-batches
            wall_clock: time took for testing
        """
        results = {}
        eval_data = None
        if not self.train_error:
            tab = " "
            if mode == 'default':
                results, eval_data = self.run_epoch(dataset, k=k, training=False, batch_wise_eval=batch_wise_eval)
            elif mode == 'apr':
                results, eval_data = self.run_adv_epoch(dataset, k=k, training=False, batch_wise_eval=batch_wise_eval)
            elif mode == 'rl':
                results, eval_data = self.run_discriminator_epoch(dataset, k=k, training=False, min_steps=min_steps,
                                                                  min_sessions=min_sessions,
                                                                  batch_wise_eval=batch_wise_eval)
            elif mode == 'rl-split':
                results, eval_data = self.run_discriminator_epoch(dataset, k=k, training=False, min_steps=min_steps,
                                                                  min_sessions=min_sessions,
                                                                  batch_wise_eval=batch_wise_eval)
            elif mode == 'rl-updates':
                results, eval_data = self.run_discriminator_epoch(dataset, k=k, training=False, min_steps=min_steps,
                                                                  min_sessions=min_sessions,
                                                                  batch_wise_eval=batch_wise_eval)
            else:
                raise AssertionError('Testing mode not declared')
            out_results = [f'{k}:{v}' for k, v in results.items()]
            print(f'Test :  {tab}{tab.join(out_results)}')

        else:
            raise AssertionError('Error found in trained model')
        return results, eval_data

    def print_parameters(self):
        for name, param in self.hgru.named_parameters():
            # print(name)
            print("name:{:30.30} norm: {:5f}".format(name, torch.norm(param).item()))
            # print('max_abs :{}'.format(param.abs().max(-1)))

    def run_adv_epoch(self, dataset, k=20, training=True, eps=.5, noise_mode='grad',
                      std=.1, mean=.0, lambda_reg=0.5, batch_wise_eval=False):

        """ Run a single training epoch """
        start_time = datetime.now()
        from modules.loss import APRLoss
        self.loss_fn_adv = APRLoss(lambda_reg, self.loss_type)

        # initialize
        losses = []
        recalls = []
        mrrs = []
        eval_data = pd.DataFrame([])
        eval_cutoffs = [5, 10, 20]
        optimizer_session = Optimizer(params=self.hgru.user_gru.parameters(),
                                      optimizer_type=self.adapt,
                                      lr=self.learning_rate,
                                      weight_decay=self.decay,
                                      momentum=self.momentum,
                                      eps=self.eps)
        # optimizer_user = self.optimizer_user
        Hs, Hu = self.hgru.session_gru.init_hidden(), self.hgru.user_gru.init_hidden()
        # if not training:
        #       self.hgru.eval()

        device = self.device

        # Start the training loop
        loader = SessionDataLoader(dataset, batch_size=self.batch_size)

        for input, target, Sstart, Ustart, in_eval in loader:

            input = input.to(device)
            target = target.to(device)
            Sstart = torch.Tensor(Sstart[:, None]).to(device)
            Ustart = torch.Tensor(Ustart[:, None]).to(device)

            self.static_model.zero_grad()
            optimizer_session.zero_grad()

            if training:
                logit, Hu, Hs = self.static_model(input, Sstart, Ustart, Hs.detach(), Hu.detach())
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_fn(logit_sampled)
                loss.backward()
                noise = self.static_model.create_adversarial(mode=noise_mode, eps=eps, mean=mean, std=std)

                ##import pdb
                # pdb.set_trace()

                # Adversarial regularization
                self.hgru.zero_grad()
                optimizer_session.zero_grad()
                logit_adv, Hu_adv, Hs_adv = self.hgru.adv_forward(input, Sstart, Ustart, Hs.detach(), Hu.detach(),
                                                                  noise=noise)
                logit_sampled_adv = logit_adv[:, target.view(-1)]
                # self.hgru.zero_grad()
                # self.optimizer_session.zero_grad()

                loss_adv = self.loss_fn_adv(logit_sampled.detach(), logit_sampled_adv)
                losses.append(loss_adv.item())
            else:
                noise = self.hgru.create_adversarial(mode='zeros')
                logit, Hu, Hs = self.hgru.adv_forward(input, Sstart, Ustart, Hs.detach(), Hu.detach(),
                                                      noise=noise)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_fn(logit_sampled)
                losses.append(loss.item())
            if torch.isnan(loss):
                self.print_parameters()
                import pdb
                pdb.set_trace()
                self.train_error = True
                raise RuntimeError("nan Loss")
                break

            eval_data = self.evaluate(batch_wise_eval, eval_cutoffs, eval_data, in_eval, loss, logit, mrrs, recalls,
                                      target,
                                      training)

            if training:

                loss_adv.backward()

                if self.clip_grad != -1:
                    _ = torch.nn.utils.clip_grad_value_(self.hgru.parameters(), self.clip_grad)
                    for p in self.hgru.parameters():
                        import pdb
                        # pdb.set_trace()
                        if p.grad is not None:
                            p.data.add_(self.learning_rate * p.grad)

                optimizer_session.step()
                optimizer_session.zero_grad()

                # optimizer_user.step()
                # flush the gradient after the optimization
                # optimizer_user.zero_grad()  # flush the gradient after the optimization

        eval_data, results = self.build_results(batch_wise_eval, eval_cutoffs, eval_data, losses, mrrs, recalls,
                                                start_time)
        return results, eval_data

    def run_discriminator_epoch(self, dataset, k=20, training=True, min_steps=3, min_sessions=3, strategy='replace',
                                batch_wise_eval=False):

        """ Run a single training epoch """
        start_time = datetime.now()

        # initialize
        losses = []
        recalls = []
        mrrs = []
        eval_data = pd.DataFrame([])
        eval_cutoffs = [5, 10, 20]
        optimizer_session = self.optimizer_session
        # optimizer_user = self.optimizer_user
        Hs, Hu = self.hgru.session_gru.init_hidden(), self.hgru.user_gru.init_hidden()

        device = self.device
        steps = torch.ones(self.batch_size, 1).to(device)
        session_count = torch.ones(self.batch_size, 1).to(device)
        session_steps = torch.ones(self.batch_size, 1).to(device)
        # Start the training loop
        loader = SessionDataLoader(dataset, batch_size=self.batch_size)
        count = 0
        self.print_parameters()
        for input, target, Sstart, Ustart, in_eval in loader:

            input = input.to(device)
            target = target.to(device)
            Sstart = torch.Tensor(Sstart[:, None]).to(device)
            Ustart = torch.Tensor(Ustart[:, None]).to(device)

            # steps *= (1.0 - Sstart)
            # steps += 1.0
            # session_steps =  session_steps * (1.0 - Sstart)+ (1.0 - Sstart)
            # mask_for_interactions = (steps % min_steps > 0).float()
            from modules.ddpg import generate_interaction_mask

            steps, session_count, mask_for_interactions = generate_interaction_mask(steps, session_count, Sstart,
                                                                                    Ustart,
                                                                                    min_steps, min_sessions,
                                                                                    torch.ones_like(steps).cuda())
            H_s_init = (Hs[-1] * Sstart) * (1 - Ustart)

            logit, Hu, Hs = self.hgru.forward_discriminator(input, Sstart, Ustart, Hs.detach(), Hu.detach(),
                                                            H_s_init.detach(),
                                                            mask_for_interactions, steps)
            # H_s_init = next_H_s_init * (1.0 - Sstart) + new_H_s_init * Sstart

            if torch.isnan(logit.mean()):
                raise AssertionError("nan output")
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_fn(logit_sampled)
            losses.append(loss.item())
            eval_data = self.evaluate(batch_wise_eval, eval_cutoffs, eval_data, in_eval, loss, logit, mrrs, recalls,
                                      target,
                                      training)

            if training:
                loss.backward()

                if self.clip_grad != -1:
                    _ = torch.nn.utils.clip_grad_value_(self.hgru.parameters(), self.clip_grad)
                    for p in self.hgru.parameters():
                        import pdb
                        # pdb.set_trace()
                        if p.grad is not None:
                            p.data.add_(self.learning_rate * p.grad)
                optimizer_session.step()
                # self.print_parameters()
                optimizer_session.zero_grad()

                # optimizer_user.step()
                # flush the gradient after the optimization
                # optimizer_user.zero_grad()  # flush the gradient after the optimization

        eval_data, results = self.build_results(batch_wise_eval, eval_cutoffs, eval_data, losses, mrrs, recalls,
                                                start_time)
        return results, eval_data

