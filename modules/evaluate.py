import torch
from scipy.special.cython_special import logit


def get_recall(indices, targets, batch_wise=False):
    """ Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
        batch_wise (Bool)
    Returns:
        recall (float): the recall score
    """
    targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
    hits = (targets == indices).nonzero()
    if batch_wise:
        return ((targets == indices) * 1.0).sum(dim=-1).view(-1, 1)
    else:
        if len(hits) == 0: return 0
        recall = (targets == indices).nonzero().size(0) / targets.size(0)

        return recall


def get_mrr(indices, targets, batch_wise=False):
    """ Calculates the MRR score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
        batch_wise (Bool)
    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1, 1).expand_as(indices)
    # ranks of the targets, if it appears in your indices
    hits = (targets == indices).nonzero()

    if len(hits) == 0:
        if batch_wise:
            return torch.zeros(targets.shape[0], 1).cuda()
        else:
            return 0

    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    if batch_wise:
        import pdb
        # pdb.set_trace()
        buffer = torch.zeros(targets.shape[0]).cuda()
        if len(hits) > 0:
            buffer[hits[:, 0]] = torch.reciprocal(ranks)
        buffer = buffer.view(-1, 1)
        return buffer
    rranks = torch.reciprocal(ranks)  # reciprocal ranks

    mrr = torch.sum(rranks) / targets.size(0)  # / targets.size(0)

    return mrr.item()


def evaluate(logits, targets, k=20, batch_wise=False):
    """ Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(logits, k, -1)
    recall = get_recall(indices, targets, batch_wise)
    mrr = get_mrr(indices, targets, batch_wise)

    return recall, mrr


def evaluate_with_ranks(logits, targets, k=20, batch_wise=False):
    logits_t = logits.t()
    ranks = (logits_t > logits_t[targets].diag()).sum(0) + 1
    mrr, recall = get_metrics_from_ranks(batch_wise, k, ranks)
    return recall, mrr, ranks


def get_metrics_from_ranks(batch_wise, k, ranks):
    ranks_ok = (ranks <= k)
    if batch_wise:
        recall = ranks_ok.float().view(-1, 1)
        mrr = (ranks_ok.float() / ranks.float()).view(-1, 1)
    else:
        recall = ranks_ok.float().mean()
        mrr = (ranks_ok.float() / ranks.float()).mean()
    return mrr, recall


def evaluate_multiple_with_ranks(logits, targets, eval_cutoffs=[5, 10, 20], batch_wise=False):
    logits_t = logits.t()
    ranks = (logits_t > logits_t[targets].diag()).sum(0) + 1
    recall, mrr = [], []
    for k in eval_cutoffs:
        mrr_k, recall_k = get_metrics_from_ranks(batch_wise, k, ranks)
        recall.append(recall_k)
        mrr.append(mrr_k)
    return recall, mrr, ranks


def evaluate_multiple(logits, targets, eval_cutoffs=[5, 10, 20], batch_wise=False):
    """ Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(logits, max(eval_cutoffs), -1)
    recall, mrr = [], []
    for k in eval_cutoffs:
        indices_k = indices[:, :k]
        targets_k = targets
        recall_k, mrr_k = get_recall(indices_k, targets_k, batch_wise), get_mrr(indices_k, targets_k, batch_wise)

        recall.append(recall_k)

        mrr.append(mrr_k)
    # print([[str(x.size()) for x in recall], str(targets.size()), str(indices_k.size())])
    return recall, mrr


# Test
# torch.random.manual_seed(0)
#B, C, K = 5, 100, 5
#logits = torch.rand(B, C).cuda()
#targets = torch.randint(C, (B,)).cuda()
#print(targets)

# evaluate_with_ranks(logits, targets,K,False)
# evaluate_with_ranks(logits, targets,K,True)
#print(torch.cat(evaluate_multiple_with_ranks(logits, targets,batch_wise=True)[0],-1))
#print(torch.cat(evaluate_multiple(logits, targets,batch_wise=True)[0],-1))
