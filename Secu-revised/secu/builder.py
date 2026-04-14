# Copyright (c) Alibaba Group
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from config import clusters_amount


class SeCu(nn.Module):
    """
    Build a SeCu model with multiple clustering heads
    """

    def __init__(self, base_encoder, K, dim=128, num_ins=50000, tx=0.05, tw=0.05, alpha=6000, dual_lr=0.1,
                 lratio=0.9, constraint='size'):
        super(SeCu, self).__init__()
        self.K = K
        self.tx = tx
        self.tw = tw
        self.num_head = len(self.K)
        self.alpha = alpha
        self.dual_lr = dual_lr
        self.lratio = lratio
        self.lbound = [lratio / curK for curK in self.K]
        self.cst = constraint
        # create the encoder with projection head
        self.encoder = base_encoder(num_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim))
            
        # list for cluster assignments
        self.assign_labels = []
        self.counters = []
        for i in range(0, self.num_head):
            self.register_buffer("assign_labels_" + str(i), torch.ones(num_ins, dtype=torch.long))
            self.register_parameter("center_" + str(i), Parameter(F.normalize(torch.randn(dim, self.K[i]), dim=0))) 
            self.register_buffer("counter_" + str(i), torch.zeros(self.K[i]))
        if self.cst == 'size' or self.cst == 'size-mml':
            self.lduals = []
            for i in range(0, self.num_head):
                self.register_buffer("ldual_" + str(i), torch.zeros(self.K[i]))

    @torch.no_grad()
    def load_param(self):
        for i in range(0, self.num_head):
            self.assign_labels.append(getattr(self, "assign_labels_" + str(i)))
            self.counters.append(getattr(self, "counter_" + str(i)))
        if self.cst == 'size' or self.cst == 'size-mml':
            for i in range(0, self.num_head):
                self.lduals.append(getattr(self, "ldual_" + str(i)))

    @torch.no_grad()
    def gen_label_size(self, obj_val, branch):
        return torch.argmin(obj_val - self.lduals[branch], dim=1).squeeze(-1)

    @torch.no_grad()
    def gen_label_entropy(self, obj_val, labels, branch, epoch):
        cur_count = self.counters[branch]
        if epoch == 0:
            base = torch.sum(cur_count)
            if base == 0:
                return torch.argmin(obj_val, dim=1).squeeze(-1)
            tmp_prob = cur_count / (base + 1.)
            if torch.min(tmp_prob) > 0:
                tmp_entropy = tmp_prob * torch.log(tmp_prob)
            else:
                tmp_entropy = torch.zeros(tmp_prob.shape, device=tmp_prob.device)
                tmp_entropy[tmp_prob>0] = tmp_prob[tmp_prob>0] * torch.log(tmp_prob[tmp_prob>0])
            tmp_prob_increase = tmp_prob + 1. / (base + 1.)
            tmp_entropy_increase = tmp_prob_increase * torch.log(tmp_prob_increase)
            entropy = tmp_entropy_increase - tmp_entropy
            return torch.argmin(obj_val + self.alpha * entropy.repeat(obj_val.shape[0], 1), dim=1).squeeze(-1)
        else:
            base = torch.sum(cur_count)
            tmp_prob = cur_count / base
            if torch.min(tmp_prob) > 0:
                tmp_entropy = tmp_prob * torch.log(tmp_prob)
            else:
                tmp_entropy = torch.zeros(tmp_prob.shape, device=tmp_prob.device)
                tmp_entropy[tmp_prob>0] = tmp_prob[tmp_prob>0] * torch.log(tmp_prob[tmp_prob>0])
            tmp_prob_increase = tmp_prob + 1. / base
            tmp_entropy_increase = tmp_prob_increase * torch.log(tmp_prob_increase)
            label_prob = tmp_prob[labels]
            label_entropy = label_prob * torch.log(label_prob)
            new_prob = label_prob - 1. / base
            if torch.min(new_prob) > 0:
                new_entropy = new_prob * torch.log(new_prob)
            else:
                new_entropy = torch.zeros(new_prob.shape, device=new_prob.device)
                new_entropy[new_prob>0] = new_prob[new_prob>0] * torch.log(new_prob[new_prob>0])
            entropy_reset = (new_entropy - label_entropy).reshape(-1, 1)
            entropy = entropy_reset - tmp_entropy + tmp_entropy_increase
            entropy.scatter_(1, labels.reshape(-1, 1), 0)
            return torch.argmin(obj_val + self.alpha * entropy, dim=1).squeeze(-1)

    @torch.no_grad()
    def update_label(self, targets, labels, branch):
        self.assign_labels[branch][targets] = labels

    @torch.no_grad()
    def get_label(self, target, branch):
        return self.assign_labels[branch][target]

    @torch.no_grad()
    def reset_count(self):
        for i in range(0, self.num_head):
            self.counters[i] *= 0

    @torch.no_grad()
    def update_count(self, labels, last_labels, branch, epoch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        if epoch == 0:
            self.counters[branch][label_idx] += label_count
        else:
            last_label_idx, last_label_count = torch.unique(last_labels, return_counts=True)
            self.counters[branch][label_idx] += label_count
            self.counters[branch][last_label_idx] -= last_label_count

    @torch.no_grad()
    def update_dual_mini_batch(self, labels, branch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        self.lduals[branch][label_idx] -= self.dual_lr / len(labels) * label_count
        self.lduals[branch] += self.dual_lr * self.lbound[branch]
        if self.lratio < 1:
            self.lduals[branch][self.lduals[branch] < 0] = 0
        self.counters[branch][label_idx] += label_count

    @torch.no_grad()
    def get_centers(self):
        centers = []
        for i in range(0, self.num_head):
            centers.append(F.normalize(getattr(self, "center_" + str(i)).clone().detach(), dim=0))
        return centers

    @torch.no_grad()
    def get_pred(self,x):
        x1 = self.encoder(x)
        x1_proj = F.normalize(x1, dim=1)
        head_idx =  clusters_amount-self.K[0]  #修改
        cur_c = F.normalize(getattr(self, "center_" + str(head_idx)), dim=0)
        proj_c1 = x1_proj @ cur_c
        # print(proj_c1.shape)
        return proj_c1
    
    @torch.no_grad()
    def get_feature(self,x):
        x1 = self.encoder(x)
        x1_proj = F.normalize(x1, dim=1)
        return x1_proj
    # ------------------------------------------------------------
# ❶ 分散式 gather 工具
    @staticmethod
    @torch.no_grad()
    def _gather_all(tensor: torch.Tensor) -> torch.Tensor:
        """Collect tensor from all GPUs (DDP safe)."""
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        out = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(out, tensor, async_op=False)
        return torch.cat(out, dim=0)

    # ❷ 用「簇內 cos-sim 總和最大樣本」覆寫中心
    @torch.no_grad()
    def recompute_medoids(self, loader):
        """
        Re-estimate each center as the medoid (cos-sim sum argmax) of its cluster.
        """
        self.eval()
        dev = next(self.parameters()).device

        feats_all = []
        labels_all = [ [] for _ in range(self.num_head) ]

        for (views, idx) in loader:                # 只用第一視角即可
            v1 = views[0].to(dev, non_blocking=True)
            f  = F.normalize(self.encoder(v1), dim=1).cpu()
            feats_all.append(f)
            for h in range(self.num_head):
                labels_all[h].append(self.assign_labels[h][idx].cpu())

        #feats_all = torch.cat(self._gather_all(torch.cat(feats_all)), dim=0)
        feats_all = self._gather_all(torch.cat(feats_all, dim=0))
        for h in range(self.num_head):
            #labels_all[h] = torch.cat(self._gather_all(torch.cat(labels_all[h])), dim=0)
            labels_all[h] = self._gather_all(torch.cat(labels_all[h], dim=0))

        for h, K in enumerate(self.K):
            center_h = getattr(self, f"center_{h}")          # Parameter or buffer
            for k in range(K):
                mask = labels_all[h] == k
                if not mask.any():          # 避免空簇
                    continue
                sub = feats_all[mask]       # n_k × dim  (全部保留)
                # --- 方法 A：完整 O(n_k²) cos-sim ---
                sim_mat = sub @ sub.T       # n_k × n_k
                idx = sim_mat.sum(dim=1).argmax()
                new_c = F.normalize(sub[idx], dim=0)
                center_h.data[:, k] = new_c

    @torch.no_grad()
    def recompute_medoids_topk(self, loader, topk_ratio=0.9):
        """
        Re-estimate each center as the medoid (cos-sim sum argmax) of its cluster,
        but only use top-k% samples (with highest similarity to current center) for robustness.
        """
        self.eval()
        dev = next(self.parameters()).device

        feats_all = []
        labels_all = [ [] for _ in range(self.num_head) ]

        for (views, idx) in loader:
            v1 = views[0].to(dev, non_blocking=True)
            f = F.normalize(self.encoder(v1), dim=1).cpu()
            feats_all.append(f)
            for h in range(self.num_head):
                labels_all[h].append(self.assign_labels[h][idx].cpu())

        feats_all = self._gather_all(torch.cat(feats_all, dim=0))
        for h in range(self.num_head):
            labels_all[h] = self._gather_all(torch.cat(labels_all[h], dim=0))

        for h, K in enumerate(self.K):
            center_h = getattr(self, f"center_{h}")  # [dim, K]
            for k in range(K):
                mask = labels_all[h] == k
                if not mask.any():
                    continue

                sub = feats_all[mask]  # [n_k, dim]
                if len(sub) < 2:
                    continue

                center_k = F.normalize(center_h[:, k].unsqueeze(0), dim=1).cpu()  # [1, dim] on CPU
                sims = F.cosine_similarity(sub, center_k, dim=1)  # [n_k]

                # 取前 topk_ratio% 相似度高的樣本
                topk = int(len(sims) * topk_ratio)
                if topk < 1:
                    continue
                _, indices = torch.topk(sims, topk)
                sub_top = sub[indices]

                # 找 medoid：總相似度最大的樣本
                sim_mat = sub_top @ sub_top.T  # [topk, topk]
                idx = sim_mat.sum(dim=1).argmax()
                new_c = F.normalize(sub_top[idx], dim=0)
                center_h.data[:, k] = new_c

    '''def forward(self, view1, view2, pre_centers, target, epoch, criterion, args):
        x1 = self.encoder(view1)
        x1_proj = F.normalize(x1, dim=1)
        x2 = self.encoder(view2)
        x2_proj = F.normalize(x2, dim=1)
        loss_proj_x = 0
        loss_proj_c = 0
        idx = torch.arange(len(target), device=target.device)
        targets = concat_all_gather(target)
        for i in range(0, self.num_head):
            cur_c = F.normalize(getattr(self, "center_" + str(i)), dim=0)
            proj_c1 = x1_proj.clone().detach() @ cur_c
            proj_c2 = x2_proj.clone().detach() @ cur_c
            with torch.no_grad():
                pre_c = pre_centers[i]
                # generate cluster assignments
                obj_val = -0.5 * (proj_c1 + proj_c2)
                if epoch == 0:
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, None, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, None, i, epoch)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)
                    cur_label = self.get_label(target, i)
                else:
                    cur_label = self.get_label(target, i)
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, cur_label, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, self.get_label(targets, i), i, epoch)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)

            # loss for cluster centers
            with torch.no_grad():
                logits_proj_c1 = proj_c1.clone().detach()
                logits_proj_c2 = proj_c2.clone().detach()
            logits_proj_c1[idx, label] = proj_c1[idx, label]
            logits_proj_c2[idx, label] = proj_c2[idx, label]
            loss_proj_c += criterion(logits_proj_c1 / self.tw, label) + criterion(logits_proj_c2 / self.tw, label)

            # loss for representations
            proj_x1 = x1_proj @ pre_c / self.tx
            proj_x2 = x2_proj @ pre_c / self.tx
            with torch.no_grad():
                soft_label_view1 = (1.-args.secu_tau) * F.softmax(proj_x2, dim=1)
                soft_label_view2 = (1.-args.secu_tau) * F.softmax(proj_x1, dim=1)
                soft_label_view1[idx, cur_label] += args.secu_tau
                soft_label_view2[idx, cur_label] += args.secu_tau
            loss_proj_x -= (torch.mean(torch.sum(F.log_softmax(proj_x1, dim=1) * soft_label_view1, dim=1)) +
                torch.mean(torch.sum(F.log_softmax(proj_x2, dim=1) * soft_label_view2, dim=1)))

        loss_x = loss_proj_x / (2. * self.num_head)
        loss_c = loss_proj_c / (2. * self.num_head)
        return loss_x, loss_c'''
    
    
    def forward(self, view1, view2, pre_centers, target, epoch, criterion, args):
        x1 = self.encoder(view1)
        x1_proj = F.normalize(x1, dim=1)
        x2 = self.encoder(view2)
        x2_proj = F.normalize(x2, dim=1)
        loss_proj_x = 0
        loss_proj_c = 0
        #add
        mml_loss_total = 0
        idx = torch.arange(len(target), device=target.device)
        targets = concat_all_gather(target)
        for i in range(0, self.num_head):
            cur_c = F.normalize(getattr(self, "center_" + str(i)), dim=0)
            proj_c1 = x1_proj.clone().detach() @ cur_c
            proj_c2 = x2_proj.clone().detach() @ cur_c
            with torch.no_grad():
                pre_c = pre_centers[i]
                obj_val = -0.5 * (proj_c1 + proj_c2)
                if epoch == 0:
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, None, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, None, i, epoch)
                    elif self.cst == 'size-mml':
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)
                    cur_label = self.get_label(target, i)
                else:
                    cur_label = self.get_label(target, i)
                    if self.cst == 'entropy':
                        label = self.gen_label_entropy(obj_val, cur_label, i, epoch)
                        labels = concat_all_gather(label)
                        self.update_count(labels, self.get_label(targets, i), i, epoch)
                    elif self.cst == 'size-mml':
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    else:
                        label = self.gen_label_size(obj_val, i)
                        labels = concat_all_gather(label)
                        self.update_dual_mini_batch(labels, i)
                    self.update_label(targets, labels, i)

            with torch.no_grad():
                logits_proj_c1 = proj_c1.clone().detach()
                logits_proj_c2 = proj_c2.clone().detach()
            logits_proj_c1[idx, label] = proj_c1[idx, label]
            logits_proj_c2[idx, label] = proj_c2[idx, label]
            loss_proj_c += criterion(logits_proj_c1 / self.tw, label) + criterion(logits_proj_c2 / self.tw, label)

            proj_x1 = x1_proj @ pre_c / self.tx
            proj_x2 = x2_proj @ pre_c / self.tx
            with torch.no_grad():
                soft_label_view1 = (1.-args.secu_tau) * F.softmax(proj_x2, dim=1)
                soft_label_view2 = (1.-args.secu_tau) * F.softmax(proj_x1, dim=1)
                soft_label_view1[idx, cur_label] += args.secu_tau
                soft_label_view2[idx, cur_label] += args.secu_tau
            #org swap loss cal
            loss_proj_x -= (torch.mean(torch.sum(F.log_softmax(proj_x1, dim=1) * soft_label_view1, dim=1)) +
                            torch.mean(torch.sum(F.log_softmax(proj_x2, dim=1) * soft_label_view2, dim=1)))
            
            if self.cst == 'size-mml' and epoch >= 100:
                mml_fn = GraphModularityLoss(epsilon=0.8, min_k=1)
                with torch.no_grad():
                    x_proj_all = concat_all_gather(x1_proj)
                    label_all = concat_all_gather(cur_label)
                mml_loss = mml_fn(x_proj_all, label_all)
                mml_loss_total += mml_loss
        loss_x = loss_proj_x / (2. * self.num_head)
        loss_c = loss_proj_c / (2. * self.num_head)
        mml_loss_total = mml_loss_total / self.num_head

        if self.cst == 'size-mml' and epoch >= 100:
            #loss_c =  loss_size   #  size & self_correct
            mml_loss_total = 12 * mml_loss_total
        else:
            #loss_c = loss_size
            mml_loss_total = torch.tensor(0.0, device=x1.device)
        return loss_x, loss_c,mml_loss_total

# utils
'''@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output'''
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Gathers tensor from all GPUs if in distributed mode.
    Otherwise, returns the input tensor directly.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor  # single GPU 

    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return torch.cat(tensors_gather, dim=0)

class GraphModularityLoss(nn.Module):
    def __init__(self, epsilon=0.8, min_k=1):
        super().__init__()
        self.epsilon = epsilon
        self.min_k = min_k

    '''def build_affinity_graph(self, x_proj):
        sim = F.cosine_similarity(x_proj.unsqueeze(1), x_proj.unsqueeze(0), dim=-1)  # [N, N]
        sim.fill_diagonal_(-float('inf'))
        A = torch.zeros_like(sim)
        A[sim > self.epsilon] = sim[sim > self.epsilon]

        for i in range(A.size(0)):
            if A[i].sum() == 0:
                topk = sim[i].topk(self.min_k).indices
                A[i, topk] = sim[i, topk]

        return torch.max(A, A.T)'''
    def build_affinity_graph(self, x_proj):
        sim = F.cosine_similarity(x_proj.unsqueeze(1), x_proj.unsqueeze(0), dim=-1)
        sim.fill_diagonal_(-float('inf'))
        
        A = torch.zeros_like(sim)
        
        # 1. 主要門檻：大於 0.8 的直接連
        A[sim > self.epsilon] = sim[sim > self.epsilon]

        # 2. 保底機制：修改為「有條件的強制連線」
        # 你的想法：如果主要門檻沒過，但相似度還不錯 (> 0.7)，才給它連線
        rescue_threshold = 0.6  # 你設定的保底門檻
        
        for i in range(A.size(0)):
            if A[i].sum() == 0:  # 變成了孤立點
                # 找出最像的那一個 (Top-1)
                best_val, best_idx = sim[i].topk(1)
                
                # ★ 關鍵修改：只有當最像的這個人大於 0.7 才連
                if best_val > rescue_threshold:
                    A[i, best_idx] = best_val
                # else: 就讓它繼續當孤立點 (寧缺勿濫)

        return torch.max(A, A.T)

    def compute_modularity_matrix(self, A):
        d = A.sum(dim=1, keepdim=True)      # [N, 1]
        m = d.sum() / 2                     # scalar
        B = A - (d @ d.T) / (2 * m + 1e-6)  # [N, N]
        return B, m

    def normalize_U(self, U):
        sqrt_U = torch.sqrt(U)
        norm_factor = sqrt_U.sum()
        return sqrt_U * math.sqrt(U.size(0)) / (norm_factor + 1e-6)

    def forward(self, x_proj, pseudo_labels):
        A = self.build_affinity_graph(x_proj)  # [N, N]
        B, m = self.compute_modularity_matrix(A)
        U = F.one_hot(pseudo_labels, num_classes=pseudo_labels.max().item() + 1).float()
        U_tilde = self.normalize_U(U)
        loss = - torch.trace(U_tilde.T @ B @ U_tilde) / (2 * m + 1e-6)
        return loss
