# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 11:05
@File    : adversarial_training.py
@Function: XX
@Other: XX
"""
import torch


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD(object):

    def __init__(self, model, emb_name, epsilon=1.0, alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()  # 第一次进来时备份原始的embedding参数
                if param.grad is None:
                    norm = 0
                else:
                    norm = torch.norm(param.grad)  # 计算矩阵的Frobenius norm  (Frobenius 范数)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm  # 把0.3均匀分配到 embedding层的梯度
                    # 函数加了下划线的属于内建函数，将要改变原来的值，没有加下划线的并不会改变原来的数据，引用时需要另外赋值给其他变量
                    param.data.add_(r_at)  # 这里换句话说对embedding加了总和0.3的扰动(也可以理解为单独更新了embedding层的参数)
                    param.data = self.project(name, param.data, self.epsilon)  # 重新规范扰动的整体情况 防止一次次加下去偏离太多

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        """
        重新归一化扰动，使扰动控制在epsilon内
        :param param_name: 参数名
        :param param_data: 加过扰动的embedding矩阵
        :param epsilon: 超参 默认1.0
        :return:
        """
        r = param_data - self.emb_backup[param_name]  # 减回去 r不就是扰动吗 这里其实是和原始embedding比较！
        if torch.norm(r) > epsilon:  # 正常情况下第一次应该就等于alpha 三次也应该就是小于0.9！
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_backup:
                    param.grad = self.grad_backup[name]
                else:
                    param.grad = None


class FreeLB(object):

    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss


# # 使用范例说明
# fgm = FGM(model)
# for batch_input, batch_label in data:
#     # 正常训练
#     loss = model(batch_input, batch_label)
#     loss.backward() # 反向传播，得到正常的grad
#     # 对抗训练
#     fgm.attack() # 在embedding上添加对抗扰动
#     loss_adv = model(batch_input, batch_label)
#     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#     fgm.restore() # 恢复embedding参数
#     # 梯度下降，更新参数
#     optimizer.step()
#     model.zero_grad()

# pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
# K = 3
# for batch_input, batch_label in processor:
#     # 正常训练
#     loss = model(batch_input, batch_label)
#     loss.backward() # 反向传播，得到正常的grad
#     pgd.backup_grad()
#     # 对抗训练
#     for t in range(K):
#         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.processor
#         if t != K-1:
#             model.zero_grad()
#         else:
#             pgd.restore_grad()
#         loss_adv = model(batch_input, batch_label)
#         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#     pgd.restore() # 恢复embedding参数
#     # 梯度下降，更新参数
#     optimizer.step()
#     model.zero_grad()

# freelb = FreeLB()
# K = 3
# for batch_input, batch_label in processor:
#     loss = freelb.attack(model,inputs,.....)
# # https://codeantenna.com/a/rqVEKQI1Zx
