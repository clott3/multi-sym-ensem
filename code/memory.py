from asyncio import gather
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import gather_from_all


def build_mem(args):
    memory = RGBMoCo(args.feat_dim, args.mem_size, args.temp)

    return memory

class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""
    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue, k_labels, queue_labels):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)
            queue_labels.index_copy_(0, out_ids, k_labels)

    def _compute_logits_with_labels(self, z1, z2, batch_labels, queue, queue_labels, topk_labels):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
          batch_labels: labels of q
          queue_labels: labels of memory buffer
        """

        batch_labels = batch_labels.view(-1, 1)
        bsz = z1.shape[0]

        logits = (z1 @ z2.T) / self.T
        neg_logits = (z1 @ queue.T) / self.T
        
        # mask for positive
        pos_mask = torch.eq(batch_labels, batch_labels.transpose(1,0)).float().cuda(non_blocking=True)

        # mask for other view (positive) to be included in denominator
        posview_indices = torch.arange(0, bsz, dtype=torch.long).view(bsz, 1).cuda()
        posview_logits = torch.gather(logits, 1, posview_indices)

        # mask for negatives (only pick out topk from memory bank)
        neg_mask_queue = torch.eq(topk_labels.unsqueeze(-1).permute(1,0,2), queue_labels.permute(1,0))
        neg_mask_queue = neg_mask_queue.any(dim=0).float()

        neg_mask_batch = torch.eq(topk_labels.unsqueeze(-1).permute(1,0,2), batch_labels.permute(1,0))
        neg_mask_batch = neg_mask_batch.any(dim=0).float()
        denom_exp_logits = torch.exp(posview_logits) + (torch.exp(neg_logits) * neg_mask_queue).sum(dim=1, keepdim=True) + \
                           (torch.exp(logits) * neg_mask_batch).sum(dim=1, keepdim=True)

        log_prob = -1.0 * (logits - torch.log(denom_exp_logits))
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(dim=1)

        return mean_log_prob_pos.mean()


class RGBMoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(RGBMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory', torch.randn(K, n_dim))
        self.register_buffer('memory_labels', torch.empty(K, 1).fill_(-2.0))
        self.memory = F.normalize(self.memory)

    def forward(self, z1, z2, batch_labels=None, topk_labels=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """

        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        # gather all tensors across all nodes/gpus
        z1 = gather_from_all(z1)
        z2 = gather_from_all(z2)
        batch_labels = gather_from_all(batch_labels)
        topk_labels = gather_from_all(topk_labels)

        # compute logit
        queue = self.memory.clone().detach()

        if batch_labels is None or topk_labels is None:
            raise NotImplementedError('Loss not implemented with memory bank: batch_labels or topk_labels undefined')

        queue_labels = self.memory_labels.clone().detach()
        loss = self._compute_logits_with_labels(z1, z2, batch_labels, queue, queue_labels, topk_labels)

        # update memory
        self._update_memory(z2.float(), self.memory, batch_labels.view(-1, 1).float(), self.memory_labels)
        self._update_pointer(z2.size(0))

        return loss