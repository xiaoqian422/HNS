from model import objectives
# from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb

from .objectives import HardNegativeLoss


class AverageSims(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count)


class VarianceSims(object):
    """Computes and stores the Variance and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.var = 0
        self.sum = 0
        self.count = 0

    def update(self, val, avg, n=1):
        self.val = val
        self.sum += (val * n - avg) ** 2
        self.count += n
        self.var = self.sum / (self.count - 1)


class HNS(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.attn_weight = False
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        self.avg_sims_tool_pos = AverageSims()
        self.var_sims_tool_pos = VarianceSims()

        self.avg_sims_tool_neg = AverageSims()
        self.var_sims_tool_neg = VarianceSims()
        # self.queue_size = args.queue_size
        self.queue_size = 8192
        # self.momentum = args.momentum_par
        self.momentum = 0.995

        self.base_model_m, _ = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                 args.stride_size)

        self.model_pair = [self.base_model, self.base_model_m]
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(512, self.queue_size))
        self.register_buffer("text_queue", torch.randn(512, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        # x = self.base_model.encode_image(image)
        # return x[:, 0, :].float()
        x,weight = self.base_model.encode_image(image)
        # return x[:, 0, :].float()
        # # return x.float() # for CLIP ResNet visual model
        if self.attn_weight:
            return x[:,0,:].float(), weight
        else:
            return x[:,0,:].float()

    # return x.float() # for CLIP ResNet visual model
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    def encode_text(self, text):
        # x = self.base_model.encode_text(text)
        # return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        x, weight = self.base_model.encode_text(text)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        # return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        if self.attn_weight:
            return x, weight
        else:
            return x

    @torch.no_grad()
    def copy_params(self):
        for param, param_m in zip(self.model_pair[0].parameters(), self.model_pair[1].parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for param, param_m in zip(self.model_pair[0].parameters(), self.model_pair[1].parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats, idxs, index=None):
        # gather keys before updating queue
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # print(image_feats.shape[0], '--------------')
        # a=1
        # assert a>5
        assert self.queue_size % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, batch, epoch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()  # b, 512
        i_feats = F.normalize(i_feats, dim=-1)
        # i_feats = image_feats.float() # for CLIP ResNet visual model

        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()  # b, 512
        t_feats = F.normalize(t_feats, dim=-1)

        idx = batch['pids'].view(-1, 1)  # [b, 1]
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # 1, 8192+b
        pos_idx = torch.eq(idx, idx_all)
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        # label smoothing: https: // github.com / salesforce / LAVIS / blob / main / lavis / models / blip2_models / blip2_qformer.py
        # sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        with torch.no_grad():
            self._momentum_update()
            image_feats_m, text_feats_m = self.base_model_m(images, caption_ids)
            i_feats_m = image_feats_m[:, 0, :].float()
            i_feats_m = F.normalize(i_feats_m, dim=-1)

            t_feats_m = text_feats_m[torch.arange(text_feats_m.shape[0]), caption_ids.argmax(dim=-1)].float()
            t_feats_m = F.normalize(t_feats_m, dim=-1)

            i_feats_all = torch.cat([i_feats_m.t(), self.image_queue.clone().detach()], dim=1)
            t_feats_all = torch.cat([t_feats_m.t(), self.text_queue.clone().detach()], dim=1)

        sim_i2t = i_feats @ t_feats_all
        sim_t2i = t_feats @ i_feats_all

        if epoch >= 1:
            loss_hardnagetiva = HardNegativeLoss(i_feats, t_feats, sim_i2t, sim_t2i,
                                                 pos_idx, epoch,
                                                 self.avg_sims_tool_pos,
                                                 self.var_sims_tool_pos,
                                                 self.avg_sims_tool_neg,
                                                 self.var_sims_tool_neg,
                                                 self.args.nums_right_sims,
                                                 self.args.match_para,
                                                 self.args.false_margin,
                                                 self.args.false_num,
                                                 logit_scale)
            ret.update({'itc_loss': loss_hardnagetiva})
        else:
            if 'itc' in self.current_task:  # image-text contrastive (ITC) loss, InfoNCE
                loss_i2t = -torch.sum(F.log_softmax(logit_scale * sim_i2t, dim=1) * sim_targets, dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(logit_scale * sim_t2i, dim=1) * sim_targets, dim=1).mean()
                loss_ita = (loss_i2t + loss_t2i) / 2
                # ret.update({'itc_loss': objectives.compute_itc(sim_i2t, sim_t2i, sim_targets)})
                ret.update({'itc_loss': loss_ita})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:  # Cross-Modal Projection Matching Loss(CMPM)
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']
            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        self._dequeue_and_enqueue(i_feats_m, t_feats_m, idx)

        return ret


def build_model(args, num_classes=11003):
    model = HNS(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
