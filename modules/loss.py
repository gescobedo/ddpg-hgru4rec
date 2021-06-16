import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda=True):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super().__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else'cpu')
        if loss_type == 'CrossEntropy':
            self._loss_fn = SampledCrossEntropyLoss(use_cuda)
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'TOP1Max':
            self._loss_fn = TOP1MaxLoss()
        elif loss_type == 'APR':
            self._loss_fn = TOP1MaxLoss()
        elif loss_type == 'BPRMax':
            self._loss_fn = BPRMaxLoss()
        elif loss_type.startswith('BPRMax-'):
            self._loss_fn = BPRMaxLoss(bpreg=float(loss_type[len('BPRMax-'):]))
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        else:
            raise NotImplementedError

        self.to(self.device)
        self._loss_fn.to(self.device)
    def forward(self, logit):
        return self._loss_fn(logit)


class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self, use_cuda):
        """
        See Balazs Hihasi(ICLR 2016), pg.5

        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super().__init__()
        self.xe_loss = nn.CrossEntropyLoss(reduce=None)
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(0)
        target = Variable(torch.arange(batch_size).long())
        if self.use_cuda: target = target.cuda()

        return self.xe_loss(logit, target)


class BPRLoss(nn.Module):
    def __init__(self):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """

        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))

        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        # final loss
        loss = F.sigmoid(diff).mean() + F.sigmoid(logit ** 2).mean()

        return loss

class TOP1MaxLoss(nn.Module):
    def __init__(self,eps=1e-24):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()
        self.eps = eps

    def softmax_neg(self, X):
        hm = torch.ones_like(X) - torch.eye(X.shape[0]).cuda()
        X= X * hm
        X_max, _ = X.max(1)
        e_x = torch.exp(X - (X_max.unsqueeze(1))) * hm
        return e_x / e_x.sum(1).unsqueeze(1)

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # TOP1
        # ydiag = gpu_diag_wide(yhat).dimshuffle((0, 'x'))
        # return T.cast(T.mean(
        #     T.mean(T.nnet.sigmoid(-ydiag + yhat) + T.nnet.sigmoid(yhat ** 2), axis=1) - T.nnet.sigmoid(ydiag ** 2) / (
        #                M + self.n_sample)), theano.config.floatX)
        #  softmax_scores = self.softmax_neg(yhat)
        #  y = softmax_scores * (
        #              T.nnet.sigmoid(-gpu_diag_wide(yhat).dimshuffle((0, 'x')) + yhat) + T.nnet.sigmoid(yhat ** 2))
        #  return T.cast(T.mean(T.sum(y, axis=1)), theano.config.floatX)

        #softmax scores
        #hm = 1.0 - torch.eye(logit.shape[0])
        logit_in = logit - torch.diag_embed(logit.diag())
        logit_max, _ = logit_in.max(1)
        e_x = torch.exp(logit_in - (logit_max.unsqueeze(1)))
        e_x = e_x - torch.diag_embed(e_x.diag())

        softmax_scores = e_x / e_x.sum(1).unsqueeze(1)

        # differences between the item scores
        diff = -torch.diag(logit).view(-1, 1).expand_as(logit) + logit
        # final loss
        loss = torch.sum(softmax_scores * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)), 1).mean()
        #if torch.isnan(loss):
        #    import pdb
        #    pdb.set_trace()
        return loss

class BPRMaxLoss(nn.Module):
    def __init__(self, eps=1e-24, bpreg=1.0):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()
        self.eps = eps
        self.bpreg = bpreg
    def softmax_neg(self, X):
        hm = torch.ones_like(X) - torch.eye(X.shape[0]).cuda()
        X= X * hm
        X_max, _ = X.max(1)
        e_x = torch.exp(X - (X_max.unsqueeze(1))) * hm
        return e_x / e_x.sum(1).unsqueeze(1)

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # TOP1
        # ydiag = gpu_diag_wide(yhat).dimshuffle((0, 'x'))
        # return T.cast(T.mean(
        #     T.mean(T.nnet.sigmoid(-ydiag + yhat) + T.nnet.sigmoid(yhat ** 2), axis=1) - T.nnet.sigmoid(ydiag ** 2) / (
        #                M + self.n_sample)), theano.config.floatX)
        #  softmax_scores = self.softmax_neg(yhat)
        #  y = softmax_scores * (
        #              T.nnet.sigmoid(-gpu_diag_wide(yhat).dimshuffle((0, 'x')) + yhat) + T.nnet.sigmoid(yhat ** 2))
        #  return T.cast(T.mean(T.sum(y, axis=1)), theano.config.floatX)

        #softmax scores
        logit_in = logit - torch.diag_embed(logit.diag())
        logit_max, _ = logit_in.max(1)
        e_x = torch.exp(logit_in - (logit_max.unsqueeze(1)))
        e_x = e_x - torch.diag_embed(e_x.diag())

        softmax_scores = e_x / e_x.sum(1).unsqueeze(1)

        # differences between the item scores
        diff = torch.diag(logit).view(-1, 1).expand_as(logit) - logit
        # final loss
        #T.mean(-T.log(T.sum(T.nnet.sigmoid(gpu_diag_wide(yhat).dimshuffle((0,'x'))-yhat)*softmax_scores, axis=1)+1e-24)
        #                                                           +self.bpreg*T.sum((yhat**2)*softmax_scores, axis=1))
        loss = torch.sum((-F.logsigmoid(diff) + self.bpreg * logit**2) * softmax_scores, 1).mean()

       
        return loss

class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward())
    for adversial training of Generator
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward):
        """
        Inputs: pred, target, reward
            - pred: (batch_size, seq_len),
            - target : (batch_size, seq_len),
            - reward : (batch_size, ), reward of each whole sentence
        """
        one_hot = torch.zeros(pred.size(), dtype=torch.uint8)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.    data.view(-1, 1), 1)
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward.contiguous().view(-1)
        loss = -torch.sum(loss)
        return loss


class VAELoss(torch.nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal=9e-1):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), -1))
        quadratic_error = torch.pow((decoder_output - y_true_s), 2).mean()
        final = (anneal * kld) + quadratic_error
        return final

class MrrPolicyLoss(nn.Module):

    def __init__(self):
        super(MrrPolicyLoss,self).__init__()

    def forward(self, input, target):

        ret =  torch.pow((1.0-target),2) + torch.pow((1.0-input),2)

        return  ret.mean()


class APRLoss(nn.Module):
    def __init__(self,  lambda_reg=1e-3, loss='TOP1Max'):
        super(APRLoss, self).__init__()
        if loss == 'TOP1Max':
            self.loss = TOP1MaxLoss()
        if loss == 'BPRMax':
            self.loss = BPRMaxLoss()
        elif loss.startswith('BPRMax-'):
            self.loss = BPRMaxLoss(bpreg=float(loss[len('BPRMax-'):]))
        self.lambda_reg = lambda_reg

    def forward(self, yhat, adv_yhat):
        loss = self.loss(yhat) + self.lambda_reg * self.loss(adv_yhat)
        return loss


class LogLoss(nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, logits):

        return -torch.log(logits).mean()
