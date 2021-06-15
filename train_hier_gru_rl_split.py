import numpy as np
import pandas as pd
import argparse
from os import path
from datetime import datetime as dt

from modules.model import HGRU4REC as RNN
from modules.data import SessionDataset

import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('split', type=str)
parser.add_argument('session_layers', type=str)
parser.add_argument('user_layers', type=str)
parser.add_argument('--loss', type=str, default='cross-entropy')
parser.add_argument('--hidden_act', type=str, default='tanh')
parser.add_argument('--adapt', type=str, default='adagrad')
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--dropout_p_hidden_usr', type=float, default=0.0)
parser.add_argument('--dropout_p_hidden_ses', type=float, default=0.0)
parser.add_argument('--dropout_p_init', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=0.0)
parser.add_argument('--grad_cap', type=float, default=-1.0)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--lmbd', type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--init_as_normal', type=int, default=0)
parser.add_argument('--reset_after_session', type=int, default=1)
parser.add_argument('--train_random_order', type=int, default=0)
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--session_key', type=str, default='session_id')
parser.add_argument('--value_key', type=str, default='value')
parser.add_argument('--time_key', type=str, default='time')
parser.add_argument('--save_to', type=str, default='../models')
parser.add_argument('--load_from', type=str, default=None)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--rnd_seed', type=int, default=42)
parser.add_argument('--model_name', type=str, default='hgru4rec')
# sampling
parser.add_argument('--n_sample', type=int, default=0)
parser.add_argument('--sample_alpha', type=float, default=0.5)
# evaluation
parser.add_argument('--eval_cutoff', type=int, default=50)
parser.add_argument('--eval_top_pop', type=int, default=0)
parser.add_argument('--eval_boot', type=int, default=-1)
parser.add_argument('--eval_file', type=str, default=None)
# embeddings
parser.add_argument('--item_embedding', type=int, default=None)
parser.add_argument('--load_item_embeddings', type=str, default=None)
# user bias parameters 
parser.add_argument('--user_to_ses_act', type=str, default='tanh')
parser.add_argument('--user_propagation_mode', type=str, default='all')
parser.add_argument('--user_to_output', type=int, default=1)
# rl parameters
parser.add_argument('--max_frames', type=int, default=100000)
parser.add_argument('--strategy', type=str, default='replace')
parser.add_argument('--soft_tau', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--noise_mode', type=str, default='ounoise')
parser.add_argument('--action_dim', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--state_dim', type=int, default=200)
parser.add_argument('--rollout', type=int, default=3)
parser.add_argument('--min_steps', type=int, default=3)
parser.add_argument('--min_sessions', type=int, default=3)
parser.add_argument('--lr_policy', type=float, default=1e-6)
parser.add_argument('--lr_value', type=float, default=1e-6)
parser.add_argument('--split_rl_ratio', type=float, default=.5)
parser.add_argument('--policy_updates', type=int, default=1)
parser.add_argument('--rl_model_name', type=str, default=None)
parser.add_argument('--reset_buffer_limit', type=int, default=-1)
parser.add_argument('--custom_file', type=str, default='')
args = parser.parse_args()
data_home = '/media/gustavo/Storage2/Datasets'
if args.dataset == 'Tianchi-raw-full':
    prefix = 'Tianchi/raw-full'
elif args.dataset == 'Tianchi-raw-small':
    prefix = 'Tianchi/raw-small'
elif args.dataset == 'Tianchi-dense-small':
    prefix = 'Tianchi/dense-small'
elif args.dataset == 'Tianchi-dense-full':
    prefix = 'Tianchi/dense-full'
elif args.dataset == '30M-raw-full':
    prefix = '30M/raw-full'
elif args.dataset == '30M-raw-small':
    prefix = '30M/raw-small'
elif args.dataset == '30M-dense-small':
    prefix = '30M/dense-small'
elif args.dataset == '30M-dense-full':
    prefix = '30M/dense-full'
else:
    raise RuntimeError('Unknown dataset: {}'.format(args.dataset))

# Last-session-out partitioning
if args.split == 'lso-test':
    ext = 'last-session-out/sessions.hdf'
    train_key = 'train'
    test_key = 'test'
elif args.split == 'lso-valid':
    ext = 'last-session-out/sessions.hdf'
    train_key = 'valid_train'
    test_key = 'valid_test'
# Last-days-out partitioning
elif args.split == 'ldo-test':
    ext = 'last-days-out/sessions.hdf'
    train_key = 'train'
    test_key = 'test'
elif args.split == 'ldo-valid':
    ext = 'last-days-out/sessions.hdf'
    train_key = 'valid_train'
    test_key = 'valid_test'
else:
    raise RuntimeError('Unknown split: {}'.format(args.split))

sessions_path = path.join(data_home, prefix, ext)
logger.info('Loading data from: {}'.format(sessions_path))
logger.info('Split: {}'.format(args.split))
train_data_src = pd.read_hdf(sessions_path, train_key)
test_data_src = pd.read_hdf(sessions_path, test_key)

# Splitting interactions for rl-training by ratio
unique_user_sessions = train_data_src.drop_duplicates(subset=[args.user_key, args.session_key], keep='first',
                                                      inplace=False)
tail_sessions = unique_user_sessions.groupby(args.user_key).apply(
    lambda x: x.iloc[int(x[args.session_key].size * args.split_rl_ratio):])[args.session_key].values
env_test_sessions = unique_user_sessions.groupby(args.user_key).apply(
    lambda x: x.iloc[int(x[args.session_key].size * args.split_rl_ratio)])[args.session_key].values
train_data_src.loc[:, 'in_rl'] = False
train_data_src.loc[train_data_src[train_data_src[args.session_key].isin(tail_sessions)].index, 'in_rl'] = True
train_data = SessionDataset(train_data=train_data_src,
                            mode='train',
                            item_key=args.item_key,
                            user_key=args.user_key,
                            session_key=args.session_key,
                            time_key=args.time_key)

test_data = SessionDataset(train_data=train_data_src,
                           test_data=test_data_src,
                           mode='test',
                           item_key=args.item_key,
                           user_key=args.user_key,
                           session_key=args.session_key,
                           time_key=args.time_key,
                           itemmap=train_data.itemmap)
train_env_data = SessionDataset(train_data=train_data_src[train_data_src['in_rl'] == False],
                             mode='train',
                             item_key=args.item_key,
                             user_key=args.user_key,
                             session_key=args.session_key,
                             time_key=args.time_key,
                             itemmap=train_data.itemmap)
test_env_data = SessionDataset(train_data=train_data_src[train_data_src['in_rl'] == False],
                            test_data=train_data_src[train_data_src[args.session_key].isin(env_test_sessions)],
                            mode='test',
                            item_key=args.item_key,
                            user_key=args.user_key,
                            session_key=args.session_key,
                            time_key=args.time_key,
                            itemmap=train_data.itemmap)


session_layers = [int(x) for x in args.session_layers.split(',')]
user_layers = [int(x) for x in args.user_layers.split(',')]

item_embedding_values = None
if args.load_item_embeddings is not None:
    item_embedding_values = np.load(args.load_item_embeddings)

n_items = len(test_data.items)
import torch

torch.set_num_threads(10)
torch.manual_seed(args.rnd_seed)
MODELS_PATH_ENV = '../Models/hgru4rec/'
MODELS_PATH_DDPG = '../Models/hgru4rec-ddpg/'

t0 = dt.now()
logger.info('Training started')

model = RNN(input_size=n_items,
            hidden_size=session_layers[0],
            output_size=n_items,
            session_layers=session_layers,
            user_layers=user_layers,
            loss=args.loss,
            item_embedding=args.item_embedding,
            init_item_embeddings=item_embedding_values,
            hidden_act=args.hidden_act,
            dropout_p_hidden_usr=args.dropout_p_hidden_usr,
            dropout_p_hidden_ses=args.dropout_p_hidden_ses,
            dropout_p_init=args.dropout_p_init,
            lmbd=args.lmbd,
            decay=args.decay,
            grad_cap=args.grad_cap,
            sigma=args.sigma,
            adapt=args.adapt,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            init_as_normal=bool(args.init_as_normal),
            reset_after_session=bool(args.reset_after_session),
            train_random_order=bool(args.train_random_order),
            n_epochs=args.n_epochs,
            user_key=args.user_key,
            session_key=args.session_key,
            item_key=args.item_key,
            time_key=args.time_key,
            seed=args.rnd_seed,
            user_to_session_act=args.user_to_ses_act,
            user_propagation_mode=args.user_propagation_mode,
            user_to_output=bool(args.user_to_output)
            )
model_name = args.model_name + '_' + str(args.loss) + '_' + str(args.adapt) + '_' + str(
    args.learning_rate) + '_epoch' + str(args.n_epochs)
model.train(train_env_data, None, k=args.eval_cutoff, model_name=args.model_name, n_epochs=args.n_epochs, save=True, save_dir=MODELS_PATH_ENV)
del train_env_data
del test_env_data


logger.info('Training completed in {}'.format(dt.now() - t0))
# model_name = args.model_name+'_'+str(args.loss)+'_'+str(args.adapt)+'_'+str(args.learning_rate) + '_epoch' + str(args.n_epochs)
model.load_checkpoint(MODELS_PATH_ENV + model_name)
from modules.ddpg import run_ddpg, run_ddpg_with_reward_eval, DDPG
from modules.data import SessionDataLoader
from pathlib import Path

rnd=0
torch.manual_seed(0)

loader = SessionDataLoader(train_data, batch_size=args.batch_size)
loader.train_data_cols = ['in_rl']
ddpg_model = DDPG(args.state_dim, args.action_dim, args.hidden_dim, args.gamma, args.soft_tau, args.noise_mode, 
                  args.strategy, args.lr_policy, args.lr_value, args.policy_updates, args.reset_buffer_limit,
                  args.rl_model_name)

_, _, ddpg_info = run_ddpg_with_reward_eval(env=model.hgru, model=ddpg_model, loader=loader,
                                            model_dir=MODELS_PATH_DDPG, cutoff=args.eval_cutoff,
                                            name=model_name, rollout=args.rollout, batch_size=args.batch_size,
                                            max_frames=args.max_frames, min_steps=args.min_steps,
                                            min_sessions=args.min_sessions)

