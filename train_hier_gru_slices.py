import numpy as np
import pandas as pd
import argparse
from os import path
from datetime import datetime as dt
from test_utils import *
from modules.model import HGRU4REC as RNN
from modules.data import SessionDataset
import torch
import logging
import sys
import subprocess

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
parser.add_argument('--eval_slice_mode', type=str, default='user')
parser.add_argument('--mean_session_length', type=int, default=5)
parser.add_argument('--eval_type', type=str, default='default')
parser.add_argument('--upper_limit', type=int, default=-1)
parser.add_argument('--load_eval_data', type=bool, default=False)
parser.add_argument('--custom_dir', type=str, default=None)
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
parser.add_argument('--policy_updates', type=int, default=1)
parser.add_argument('--rl_model_name', type=str, default=None)
parser.add_argument('--reset_buffer_limit', type=int, default=-1)
# adversarial parameters
parser.add_argument('--eps', type=float, default=1e-24)
parser.add_argument('--neg_adv_mode', type=str, default='grad')
parser.add_argument('--std', type=float, default=5e-6)
parser.add_argument('--mean', type=float, default=.0)
parser.add_argument('--lambda_reg', type=float, default=0.01)

args = parser.parse_args()

data_home = '../Datasets'
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

session_layers = [int(x) for x in args.session_layers.split(',')]
user_layers = [int(x) for x in args.user_layers.split(',')]

item_embedding_values = None
if args.load_item_embeddings is not None:
    item_embedding_values = np.load(args.load_item_embeddings)

n_items = len(test_data.items)

torch.manual_seed(args.rnd_seed)
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
            user_to_output=bool(args.user_to_output), use_cuda=True
            )

MODELS_PATH = '../Models/hgru4rec'
MODELS_PATH_DDPG = '../Models/hgru4rec-ddpg/'
MODELS_PATH_APR = '../Models/hgru4rec-apr'

RESULTS_PATH = '../Results/hgru4rec'
RESULTS_PATH_DDPG = '../Results/hgru4rec-ddpg'
RESULTS_PATH_APR = '../Results/hgru4rec-apr'
t0 = dt.now()
model_name = args.model_name + '_' + str(args.loss) + '_' + str(args.adapt) + '_' + str(
    args.learning_rate) + '_epoch' + str(args.n_epochs)
print(model_name)
logger.info('Training started')
# model.train(train_data, None, k=args.eval_cutoff, model_name=args.model_name, n_epochs=args.n_epochs, save=True, save_dir=MODELS_PATH)
logger.info('Training completed in {}'.format(dt.now() - t0))

logger.info('Evaluation started')
t0 = dt.now()
model.load_checkpoint(MODELS_PATH + '/' + model_name)

if args.eval_type == 'rl':
    import subprocess

    results_dir = RESULTS_PATH_DDPG + '/' + args.strategy
    subprocess.call(['mkdir', '-p', results_dir])
    RESULTS_PATH = results_dir
    print('Setting up Pre-trained Policy')
    from modules.ddpg import DDPG

    ddpg_model = DDPG(args.state_dim, args.action_dim, args.hidden_dim, args.gamma, args.soft_tau, args.noise_mode,
                      args.strategy, args.lr_policy, args.lr_value, args.policy_updates, args.reset_buffer_limit,
                      args.rl_model_name)

    policy_model = ddpg_model.load_policy(MODELS_PATH_DDPG + model_name + ddpg_model.info_name)
    model.hgru.discriminator = policy_model
    print(model.hgru.discriminator.strategy)

if args.eval_type == 'rl-split':
    import subprocess

    results_dir = RESULTS_PATH_DDPG + '/' + args.strategy + '_split'
    subprocess.call(['mkdir', '-p', results_dir])
    RESULTS_PATH = results_dir
    print('Setting up Pre-trained Policy')
    from modules.ddpg import DDPG

    ddpg_model = DDPG(args.state_dim, args.action_dim, args.hidden_dim, args.gamma, args.soft_tau, args.noise_mode,
                      args.strategy, args.lr_policy, args.lr_value, args.policy_updates, args.reset_buffer_limit,
                      args.rl_model_name)

    policy_model = ddpg_model.load_policy(MODELS_PATH_DDPG + model_name + ddpg_model.info_name)
    model.hgru.discriminator = policy_model
    print(model.hgru.discriminator.strategy)

if args.eval_type == 'rl-updates':
    results_dir = RESULTS_PATH_DDPG + '/' + args.strategy + '_pupdates_' + str(args.policy_updates)
    subprocess.call(['mkdir', '-p', results_dir])
    RESULTS_PATH = results_dir
    print('Setting up Pre-trained Policy')
    from modules.ddpg import DDPG

    ddpg_model = DDPG(args.state_dim, args.action_dim, args.hidden_dim, args.gamma, args.soft_tau, args.noise_mode,
                      args.strategy, args.lr_policy, args.lr_value, args.policy_updates, args.reset_buffer_limit,
                      args.rl_model_name)

    policy_model = ddpg_model.load_policy(MODELS_PATH_DDPG + model_name + ddpg_model.info_name)
    model.hgru.discriminator = policy_model
    print(model.hgru.discriminator.strategy)
if args.eval_type == 'rl-else':
    import subprocess

    print('Setting up Pre-trained Policy')
    from modules.ddpg import DDPG

    ddpg_model = DDPG(args.state_dim, args.action_dim, args.hidden_dim, args.gamma, args.soft_tau, args.noise_mode,
                      args.strategy, args.lr_policy, args.lr_value, args.policy_updates, args.reset_buffer_limit,
                      args.rl_model_name)

    results_dir = RESULTS_PATH_DDPG + '/' + args.strategy + '_' + ddpg_model.info_name
    subprocess.call(['mkdir', '-p', results_dir])
    RESULTS_PATH = results_dir

    policy_model = ddpg_model.load_policy(MODELS_PATH_DDPG + model_name + ddpg_model.info_name)
    model.hgru.discriminator = policy_model
    print(model.hgru.discriminator.strategy)
if args.custom_dir is not None:
    results_dir = RESULTS_PATH + '/' + args.custom_dir
    subprocess.call(['mkdir', '-p', results_dir])
    RESULTS_PATH = results_dir
# In case of custom parametrization
# else:
#    raise RuntimeError('Unknown model_eval_type: {}'.format(args.eval_slice_mode))


interval_size, slice_mode, upper_limit = args.mean_session_length, args.eval_slice_mode, args.upper_limit
# interval_size = 5

#
slices_results = pd.DataFrame([])

if args.load_eval_data:
    eval_data = load_eval_data(RESULTS_PATH, args.dataset)
    slices_results = create_full_results_file_from_preds(eval_data)
else:
    full_result_dict, eval_data = model.test(test_data, mode=args.eval_type, min_steps=args.min_steps,
                                             min_sessions=args.min_sessions, batch_wise_eval=True)
    eval_data_file_name = save_eval_data(eval_data, RESULTS_PATH, args.dataset)
    full_result = {}
    for k, v in full_result_dict.items():
        full_result[k] = [v]
    full_result = pd.DataFrame.from_dict(full_result)
    full_result['slice'] = ['full']
    full_result['slice_ids_count'] = [eval_data.shape[0]]
    slices_results = slices_results.append(full_result, ignore_index=True)

multiple_mode_test(slices_results, test_data, eval_data, interval_size, upper_limit, RESULTS_PATH, args.dataset,
                   args.session_key)

print("Finished Evaluation in {}".format(dt.now() - t0))
# slices_results = pd.read_csv(results_file_url)
# print(slices_results)
