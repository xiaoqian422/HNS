import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
    # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_itc(image_features, text_features, image_features_all, text_features_all, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    image_norm_m = image_features_all / image_features_all.norm(dim=-1, keepdim=True)
    text_norm_m = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm_m
    logits_per_text = logit_scale * text_norm @ image_norm_m

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2
def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def HardNegativeLoss(im, s, sims_i2t, sims_t2i, mask, epoch, avg_sims_tool_pos, var_sims_tool_pos,
                     avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, match_para, false_margin, false_num, tau):
    diagonal_1 = sims_i2t.diag().view(sims_i2t.size(0), 1)
    d1 = diagonal_1.expand_as(sims_i2t)  # positive

    diagonal_2 = sims_t2i.diag().view(sims_t2i.size(0), 1)
    d2 = diagonal_2.expand_as(sims_t2i)

    average_sims_scores(im, s, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg)

    index_s = choice_negative(d1.detach(), sims_i2t.detach(), mask.detach(), avg_sims_tool_pos,
                              var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims,
                              match_para, false_margin, false_num)
    index_im = choice_negative(d2.detach(), sims_t2i.detach(), mask.detach(), avg_sims_tool_pos,
                               var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg,
                               nums_right_sims, match_para, false_margin, false_num)
    #
    sims_i2t_fne = sims_i2t.gather(1, index_s)
    sims_t2i_fne = sims_t2i.gather(1, index_im)

    loss_t_2_i = F.cross_entropy(sims_t2i_fne / tau, torch.zeros(sims_t2i_fne.shape[0]).to(torch.int64).cuda(),
                                 # torch.zeros：positive 放在第一个位置，所以target都为0
                                 reduction='mean')
    loss_i_2_t = F.cross_entropy(sims_i2t_fne / tau, torch.zeros(sims_i2t_fne.shape[0]).to(torch.int64).cuda(),
                                 reduction='mean')

    #     cost_s = cost_s.gather(1, index_s)
    #     cost_im = cost_im.gather(1, index_im)
    # else:
    #     cost_s = cost_s.max(1)[0]
    #     cost_im = cost_im.max(1)[0]

    return (loss_t_2_i + loss_i_2_t) / 2


def choice_negative(pos_sims, sims, mask, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg,
                    nums_right_sims, match_para, false_margin, false_num):
    # pos_sims [batch_size, 1]
    # sims [batch_size, memory_size]
    tau = 0.5  # 该参数越大，对false negative 取的权重越小
    # match_para = 10000
    # false_margin = 0.999
    num_sampled = sims.shape[1] - false_num

    neg_scores = torch.exp(-(sims - pos_sims) ** 2 * tau)
    # neg_scores = torch.ones_like(sims)
    # neg_scores = torch.zeros_like(sims)  # 初始化为0
    match_p_tmp = torch.ones_like(sims)
    if avg_sims_tool_pos.count > nums_right_sims:
        mean_data_pos = avg_sims_tool_pos.avg
        var_data_pos = var_sims_tool_pos.var

        mean_data_neg = avg_sims_tool_neg.avg
        var_data_neg = var_sims_tool_neg.var

        pos_p = normal_scores(mean_data_pos, var_data_pos, sims)
        neg_p = normal_scores(mean_data_neg, var_data_neg, sims)

        match_p = (pos_p / (pos_p + (match_para - 1) * neg_p))
        # pdb.set_trace()

        no_match_p = 1 - match_p

        match_p = torch.exp(-(match_p) * tau)

        ###
        match_mask = no_match_p < false_margin  # 0.999  极不可能为false negative，easy negative

        neg_scores[match_mask] = match_p[match_mask]
    ###

    neg_scores = neg_scores.masked_fill_(mask, 0)  # mask, 过滤掉positive

    # # margin_mask = pos_sims - margin
    # margin_mask = (sims <= (avg_sims_tool_pos.avg - margin))
    # neg_scores = neg_scores.masked_fill_(margin_mask, 0)
    # neg_scores_sum = neg_scores.sum(1)

    negative_idx = torch.multinomial(neg_scores, num_sampled)  # replacement=True 有放回采样

    negative_idx = torch.cat([torch.arange(negative_idx.shape[0]).view(-1, 1).cuda(), negative_idx],
                             dim=1)  # positive放在第一个位置

    return negative_idx


def average_sims_scores(img, txt, avgsims_pos, varsims_pos, avgsims_neg, varsims_neg):
    scores = img @ txt.T
    scores_t = scores.T

    match_index_i2t = scores.sort(1, descending=True)[1]
    match_index_t2i = scores_t.sort(1, descending=True)[1]

    # clear diagonals
    mask = torch.eye(scores.size(0)) > .5
    I = mask.cuda()
    cost_s = scores.masked_fill(I, 0)
    cost_im = scores_t.masked_fill(I, 0)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(1)[0]

    for i in range(scores.size(0)):
        if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
            avgsims_pos.update(scores[i, i])

            ###
            avgsims_neg.update(cost_s[i])
            avgsims_neg.update(cost_im[i])
    ###

    for i in range(scores.size(0)):
        if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
            varsims_pos.update(scores[i, i], avgsims_pos.avg)

            ###
            varsims_neg.update(cost_s[i], avgsims_neg.avg)
            varsims_neg.update(cost_im[i], avgsims_neg.avg)


###


def normal_scores(mean, var, x):
    normal_dist = torch.distributions.Normal(mean, torch.sqrt(var))
    output_y = normal_dist.log_prob(x).exp()

    return output_y
