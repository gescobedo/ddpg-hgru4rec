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
import torch

# torch.set_num_threads(10)
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

MODELS_PATH = '../Models/'
# model.load_state_dict(torch.load(MODELS_PATH+'last_fm_TOP1Max_Adagrad_0.1_epoch10'))
# model.eval()


# logger.info('session_layers: {}'.format(args.session_layers))
# logger.info('user_layers: {}'.format(args.user_layers))
# logger.info('loss: {}'.format(args.loss))
# logger.info('hidden_act: {}'.format(args.hidden_act))
# logger.info('batch_size: {}'.format(args.batch_size))
# logger.info('dropout_p_hidden_usr: {}'.format(args.dropout_p_hidden_usr))
# logger.info('dropout_p_hidden_ses: {}'.format(args.dropout_p_hidden_ses))
# logger.info('dropout_p_init: {}'.format(args.dropout_p_init))
# logger.info('init_as_normal: {}'.format(args.init_as_normal))
# logger.info('lmbd: {}'.format(args.lmbd))
# logger.info('grad_cap: {}'.format(args.grad_cap))
# logger.info('sigma: {}'.format(args.sigma))
# logger.info('decay (only for rmsprop): {}'.format(args.decay))
# logger.info('rnd_seed: {}'.format(args.rnd_seed))
# logger.info('')
# logger.info('TRAINING:')
# logger.info('adapt: {}'.format(model.adapt))
# logger.info('learning_rate: {}'.format(args.learning_rate))
# logger.info('momentum: {}'.format(args.momentum))
# logger.info('n_epochs: {}'.format(args.n_epochs))
# logger.info('train_random_order: {}'.format(args.train_random_order))
# logger.info('reset_after_session: {}'.format(args.reset_after_session))
# logger.info('n_epochs: {}'.format(args.n_epochs))
# logger.info('early_stopping: {}'.format(args.early_stopping))
# logger.info('save_to: {}'.format(args.save_to))
# logger.info('load_from: {}'.format(args.load_from))
# logger.info('')
# logger.info('EMBEDDINGS:')
# logger.info('item_embedding: {}'.format(args.item_embedding))
# logger.info('load_item_embeddings: {}'.format(args.load_item_embeddings))
# logger.info('')
# logger.info('USER REPR. PROPAGATION:')
# logger.info('user_to_session_act: {}'.format(args.user_to_ses_act))
# logger.info('user_propagation_mode: {}'.format(args.user_propagation_mode))
# logger.info('user_to_output: {}'.format(args.user_to_output))
# logger.info('')
# logger.info('EVALUATION:')
# logger.info('eval_cutoff: {}'.format(args.eval_cutoff))
# logger.info('eval_top_pop: {}'.format(args.eval_top_pop))
# logger.info('eval_boot: {}'.format(args.eval_boot))
# logger.info('eval_file: {}'.format(args.eval_file))

t0 = dt.now()
logger.info('Training started')
model.train(train_data, None, k=args.eval_cutoff, model_name=args.model_name, n_epochs=args.n_epochs, save=True,
            save_dir=MODELS_PATH)

logger.info('Training completed in {}'.format(dt.now() - t0))
model_name = args.model_name + '_' + str(args.loss) + '_' + str(args.adapt) + '_' + str(
    args.learning_rate) + '_epoch' + str(args.n_epochs)

#logger.info('Evaluation started')
#model.load_checkpoint(MODELS_PATH + model_name)
#full_result_dict, eval_data = model.test(test_data, mode='default', batch_wise_eval=True)
#print(full_result_dict)