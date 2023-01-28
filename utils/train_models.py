# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:04
@File    : train_models.py
@Function: XX
@Other: XX
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pynvml
from tqdm import tqdm
from utils.models import Classification, SequenceLabeling, GlobalPointerNer, GlobalPointerRe
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model, get_entity_bieos, \
    gp_entity_to_label, get_entity_gp, get_entity_gp_re
from utils.adversarial_training import PGD
from sklearn.metrics import accuracy_score, f1_score, classification_report


class TrainClassification:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.log = log
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.device = load_model_and_parallel(Classification(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(self.args.train_epochs):
            bar = tqdm(self.train_loader)
            losses = []
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss = self.criterion(train_outputs, batch_data['labels'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                loss.backward()  # 反向传播 计算当前梯度

                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 如果是最后一次对抗 恢复所有的梯度
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss_adv = self.criterion(train_outputs_adv, batch_data['labels'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                        loss_adv.backward()  # 反向传播 对抗训练的梯度 在最后一次推理的时候 叠加了一次loss
                    pgd.restore()  # 恢复embedding参数

                # 梯度裁剪 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_grad
                # 和clip_grad_value的区别在于 clip_grad_value暴力指定了区间 而clip_grad_norm做范数上的调整
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
            dev_loss, f1 = self.dev()
            if f1 > best_f1:
                best_f1 = f1
                save_model(self.args, self.model, str(epoch) + '_{:.4f}'.format(f1), self.log)
            self.log.info('[eval] epoch:{} f1_score={:.6f} best_f1_score={:.6f}'.format(epoch, f1, best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.log.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            dev_outputs = []
            dev_targets = []
            for dev_data in self.dev_loader:
                for key in dev_data.keys():
                    dev_data[key] = dev_data[key].to(self.device)
                outputs = self.model(dev_data['token_ids'],
                                     dev_data['attention_masks'],
                                     dev_data['token_type_ids'])
                loss = self.criterion(outputs, dev_data['labels'])
                tot_dev_loss += loss.detach().item()
                # flatten() 降维 默认降为1维
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(dev_data['labels'].cpu().detach().numpy().tolist())
            # accuracy = accuracy_score(dev_targets, dev_outputs)
            # 二分类 用binary 可以用pos_label指定某一类的f1
            # macro 先算每个类别的f1 再算平均 对错误的分布比较敏感
            # micro 先算总体的TP FN FP 再算f1
            # 可以这样理解 有一类比较少 但是全错了 会严重影响macro 而不会太影响micro
            micro_f1 = f1_score(dev_targets, dev_outputs, average='micro')

        return tot_dev_loss, micro_f1

    def test(self, model_path):
        model, device = load_model_and_parallel(Classification(self.args), self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            test_outputs = []
            test_targets = []
            for dev_data in tqdm(self.dev_loader):
                for key in dev_data.keys():
                    dev_data[key] = dev_data[key].to(device)
                outputs = model(dev_data['token_ids'],
                                dev_data['attention_masks'],
                                dev_data['token_type_ids'])
                loss = self.criterion(outputs, dev_data['labels'])
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(dev_data['labels'].cpu().detach().numpy().tolist())
            accuracy = accuracy_score(test_targets, test_outputs)
            micro_f1 = f1_score(test_targets, test_outputs, average='micro')
            macro_f1 = f1_score(test_targets, test_outputs, average='macro')
            self.log.info('[test] total_loss:{:.6f} accuracy={:.6f} micro_f1={:.6f} macro_f1={:.6f} '
                          .format(total_loss, accuracy, micro_f1, macro_f1))
            # report = classification_report(test_targets, test_outputs, target_names=self.labels)
            self.log.info(classification_report(test_targets, test_outputs))


class TrainSequenceLabeling:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.log = log
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.device = load_model_and_parallel(SequenceLabeling(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)

    def loss(self, model_output, true_label, attention_masks):
        if self.args.use_crf:
            loss = -self.model.crf(model_output, true_label, mask=attention_masks, reduction='mean')
        else:
            active_loss = attention_masks.view(-1) == 1
            active_logits = model_output.view(-1, model_output.size()[2])[active_loss]
            active_labels = true_label.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        return loss

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(self.args.train_epochs):  # 训练epoch数 默认50
            bar = tqdm(self.train_loader)
            losses = []
            for batch_data in bar:
                self.model.train()
                # model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
                # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss = self.loss(train_outputs, batch_data['labels'], batch_data['attention_masks'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                # loss.backward(loss.clone().detach())
                loss.backward()  # 反向传播 计算当前梯度
                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)(更新了n次embedding层)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 在对抗的最后一次恢复一开始保存的梯度 这时候的embedding参数层也加了3次扰动!
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss_adv = self.loss(train_outputs_adv, batch_data['labels'], batch_data['attention_masks'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数层

                # 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
                # optimizer.zero_grad()的作用是清除优化器涉及的所有torch.Tensor的梯度 当模型只用了一个优化器时 是等价的

            dev_loss, precision, recall, f1 = self.dev()
            if f1 > best_f1:
                best_f1 = f1
                save_model(self.args, self.model, str(epoch) + '_{:.4f}'.format(f1), self.log)
            self.log.info('[eval] epoch:{} loss:{:.6f} precision={:.6f} recall={:.6f} f1={:.6f} best_f1={:.6f}'
                          .format(epoch, dev_loss, precision, recall, f1, best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.log.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()  # 切换到测试模式 通知dropout层和batch_norm层在train和val模式间切换
        # 在eval模式下，dropout层会让所有的激活单元都通过，而batch_norm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值
        # eval模式不会影响各层的梯度计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播
        with torch.no_grad():
            # 主要是用于停止自动求导模块的工作，以起到加速和节省显存的作用
            # 具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batch_norm层的行为
            # 如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation的结果
            # with torch.no_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储gradient）
            # 从而可以更快计算，也可以跑更大的batch来测试
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in self.dev_loader:
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_outputs = self.model(dev_batch_data['token_ids'],
                                         dev_batch_data['attention_masks'],
                                         dev_batch_data['token_type_ids'],
                                         'dev')
                dev_loss = self.loss(dev_outputs, dev_batch_data['labels'], dev_batch_data['attention_masks'])
                tot_dev_loss += dev_loss.detach().item()
                if self.args.use_crf:
                    batch_output = self.model.crf.decode(dev_outputs, mask=dev_batch_data['attention_masks'])
                else:
                    batch_output = dev_outputs.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=-1)
                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity_bieos([self.id2label[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity_bieos([self.id2label[i] for i in y_true_list]))
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return tot_dev_loss, precision, recall, f1

    def test(self, model_path):
        model, device = load_model_and_parallel(SequenceLabeling(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 根据label确定有哪些实体类
        tags = [item[1] for item in self.id2label.items()]
        tags.remove('O')
        tags.remove('SEP')
        tags.remove('CLS')
        tags.remove('PAD')
        tags = [v[2:] for v in tags]
        entitys = list(set(tags))
        entitys.sort()
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                dev_outputs = model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    'dev')
                if self.args.use_crf:
                    batch_output = self.model.crf.decode(dev_outputs, mask=dev_batch_data['attention_masks'])
                else:
                    batch_output = dev_outputs.detach().cpu().numpy()
                    batch_output = np.argmax(batch_output, axis=-1)

                for y_pre, y_true in zip(batch_output, dev_batch_data['labels']):
                    R = set(get_entity_bieos([self.id2label[i] for i in y_pre]))
                    y_true_list = y_true.detach().cpu().numpy().tolist()
                    T = set(get_entity_bieos([self.id2label[i] for i in y_true_list]))
                    X_all += len(R & T)
                    Y_all += len(R)
                    Z_all += len(T)
                    for item in R & T:
                        X[entitys_to_ids[item[0]]] += 1
                    for item in R:
                        Y[entitys_to_ids[item[0]]] += 1
                    for item in T:
                        Z[entitys_to_ids[item[0]]] += 1
        len1 = max(max([len(i) for i in entitys]), 4)
        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all
        str_log = '\n{:<10}{:<15}{:<15}{:<15}\n'.format('实体' + chr(12288) * (len1 - len('实体')), 'precision', 'recall',
                                                        'f1-score')
        str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format('全部实体' + chr(12288) * (len1 - len('全部实体')), precision,
                                                                recall, f1)
        # logger.info('all_entity: precision:{:.6f}, recall:{:.6f}, f1-score:{:.6f}'
        #             .format(precision, recall, f1))
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        for entity in entitys:
            str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format(entity + chr(12288) * (len1 - len(entity)),
                                                                    precision[entitys_to_ids[entity]],
                                                                    recall[entitys_to_ids[entity]],
                                                                    f1[entitys_to_ids[entity]])
        self.log.info(str_log)


class TrainGlobalPointerNer:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.log = log
        self.criterion = MyLossNer()
        self.model, self.device = load_model_and_parallel(GlobalPointerNer(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)


    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(self.args.train_epochs):
            bar = tqdm(self.train_loader)
            losses = []
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss = self.criterion(train_outputs, batch_data['labels'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses)/len(losses)))
                loss.backward()  # 反向传播 计算当前梯度

                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 如果是最后一次对抗 恢复所有的梯度
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss_adv = self.criterion(train_outputs_adv, batch_data['labels'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                        loss_adv.backward()  # 反向传播 对抗训练的梯度 在最后一次推理的时候 叠加了一次loss
                    pgd.restore()  # 恢复embedding参数

                # 梯度裁剪 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_grad
                # 和clip_grad_value的区别在于 clip_grad_value暴力指定了区间 而clip_grad_norm做范数上的调整
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
            dev_loss, f1 = self.dev()
            if f1 > best_f1:
                best_f1 = f1
                save_model(self.args, self.model, str(epoch) + '_{:.4f}'.format(f1), self.log)
            self.log.info('[eval] epoch:{} f1_score={:.6f} best_f1_score={:.6f}'.format(epoch, f1, best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in self.dev_loader:
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_outputs = self.model(dev_batch_data['token_ids'],
                                         dev_batch_data['attention_masks'],
                                         dev_batch_data['token_type_ids'])
                dev_loss = self.criterion(dev_outputs, dev_batch_data['labels'])
                tot_dev_loss += dev_loss.detach().item()

                # dev_outputs: [batch_size, num_label, max_len, max_len]

                for y_pre, y_true in zip(dev_outputs, dev_batch_data['labels']):
                    # y_pre: [num_label, max_len, max_len]  y_true: [num_label, max_len, max_len]
                    R = set(get_entity_gp(y_pre, self.id2label))
                    y_true = y_true.detach().cpu()
                    T = set(get_entity_gp(y_true, self.id2label))
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return tot_dev_loss, f1

    def test(self, model_path):
        model, device = load_model_and_parallel(GlobalPointerNer(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 根据label确定有哪些实体类
        entitys = self.labels
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                dev_outputs = model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'])

                for y_pre, y_true in zip(dev_outputs, dev_batch_data['labels']):
                    R = set(get_entity_gp(y_pre, self.id2label))
                    y_true = y_true.detach().cpu()
                    T = set(get_entity_gp(y_true, self.id2label))
                    X_all += len(R & T)
                    Y_all += len(R)
                    Z_all += len(T)
                    for item in R & T:
                        X[entitys_to_ids[item[0]]] += 1
                    for item in R:
                        Y[entitys_to_ids[item[0]]] += 1
                    for item in T:
                        Z[entitys_to_ids[item[0]]] += 1
        len1 = max(max([len(i) for i in entitys]), 4)
        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all
        str_log = '\n{:<10}{:<15}{:<15}{:<15}\n'.format('实体' + chr(12288) * (len1 - len('实体')), 'precision', 'recall',
                                                        'f1-score')
        str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format('全部实体' + chr(12288) * (len1 - len('全部实体')), precision,
                                                                recall, f1)
        # logger.info('all_entity: precision:{:.6f}, recall:{:.6f}, f1-score:{:.6f}'
        #             .format(precision, recall, f1))
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        for entity in entitys:
            str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format(entity + chr(12288) * (len1 - len(entity)),
                                                                    precision[entitys_to_ids[entity]],
                                                                    recall[entitys_to_ids[entity]],
                                                                    f1[entitys_to_ids[entity]])
        self.log.info(str_log)


class TrainGlobalPointerRe:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.log = log
        self.criterion = MyLossRe(mask_zero=True)
        self.model, self.device = load_model_and_parallel(GlobalPointerRe(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)


    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        for epoch in range(self.args.train_epochs):
            bar = tqdm(self.train_loader)
            losses = []
            entity_losses = []
            head_losses = []
            tail_losses = []
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    if key != 'labels' and key != 'callback':
                        batch_data[key] = batch_data[key].to(self.device)
                batch_data['labels'][0] = batch_data['labels'][0].to(self.device)
                batch_data['labels'][1] = batch_data['labels'][1].to(self.device)
                batch_data['labels'][2] = batch_data['labels'][2].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'])
                loss, entity_loss, head_loss, tail_loss = self.criterion(train_outputs, batch_data['labels'])
                losses.append(loss.detach().item())
                entity_losses.append(entity_loss.detach().item())
                head_losses.append(head_loss.detach().item())
                tail_losses.append(tail_loss.detach().item())
                bar.set_postfix(loss='%.4f  ' % np.mean(losses) +
                                     '%.4f  ' % np.mean(entity_losses) +
                                     '%.4f  ' % np.mean(head_losses) +
                                     '%.4f  ' % np.mean(tail_losses))
                loss.backward()  # 反向传播 计算当前梯度

                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 如果是最后一次对抗 恢复所有的梯度
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'])
                        loss_adv, entity_loss_adv, head_loss_adv, tail_loss_adv = self.criterion(train_outputs_adv,
                                                                                                 batch_data['labels'])

                        losses.append(loss_adv.detach().item())
                        entity_losses.append(entity_loss_adv.detach().item())
                        head_losses.append(head_loss_adv.detach().item())
                        tail_losses.append(tail_loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f  ' % np.mean(losses) +
                                             '%.4f  ' % np.mean(entity_losses) +
                                             '%.4f  ' % np.mean(head_losses) +
                                             '%.4f  ' % np.mean(tail_losses))

                        # bar.set_postfix(loss='%.4f  ' % loss_adv +
                        #                      '%.4f  ' % entity_loss_adv +
                        #                      '%.4f  ' % head_loss_adv +
                        #                      '%.4f  ' % tail_loss_adv)
                        loss_adv.backward()  # 反向传播 对抗训练的梯度 在最后一次推理的时候 叠加了一次loss
                    pgd.restore()  # 恢复embedding参数

                # 梯度裁剪 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_grad
                # 和clip_grad_value的区别在于 clip_grad_value暴力指定了区间 而clip_grad_norm做范数上的调整
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
            dev_loss, f1 = self.dev()
            if f1 > best_f1:
                best_f1 = f1
                save_model(self.args, self.model, str(epoch) + '_{:.4f}'.format(f1), self.log)
            self.log.info('[eval] epoch:{} f1_score={:.6f} best_f1_score={:.6f}'.format(epoch, f1, best_f1))
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.log.info("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in self.dev_loader:
                for key in dev_batch_data.keys():
                    if key != 'labels' and key != 'callback':
                        dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_batch_data['labels'][0] = dev_batch_data['labels'][0].to(self.device)
                dev_batch_data['labels'][1] = dev_batch_data['labels'][1].to(self.device)
                dev_batch_data['labels'][2] = dev_batch_data['labels'][2].to(self.device)
                dev_outputs = self.model(dev_batch_data['token_ids'],
                                         dev_batch_data['attention_masks'],
                                         dev_batch_data['token_type_ids'])
                # dev_loss = self.criterion(dev_outputs, dev_batch_data['labels'])
                loss_dev, entity_loss_dev, head_loss_dev, tail_loss_dev = self.criterion(dev_outputs,
                                                                                         dev_batch_data['labels'])
                tot_dev_loss += loss_dev.detach().item()

                # dev_outputs: [batch_size, num_label, max_len, max_len]

                for index in range(dev_outputs[0].shape[0]):
                    # 循环每一个batch
                    R = set(get_entity_gp_re([dev_outputs[0][index], dev_outputs[1][index], dev_outputs[2][index]],
                                             dev_batch_data['attention_masks'][index].detach(),
                                             self.id2label))
                    T = set(dev_batch_data['callback'][index])
                    X += len(R & T)
                    Y += len(R)
                    Z += len(T)

            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return tot_dev_loss, f1

    def test(self, model_path):
        model, device = load_model_and_parallel(GlobalPointerRe(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 确定有哪些关系
        entitys = self.labels
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader):
                for key in dev_batch_data.keys():
                    if key != 'labels' and key != 'callback':
                        dev_batch_data[key] = dev_batch_data[key].to(device)
                dev_batch_data['labels'][0] = dev_batch_data['labels'][0].to(device)
                dev_batch_data['labels'][1] = dev_batch_data['labels'][1].to(device)
                dev_batch_data['labels'][2] = dev_batch_data['labels'][2].to(device)

                dev_outputs = model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'])

                for index in range(dev_outputs[0].shape[0]):  # batch_index
                    # 循环每一个batch
                    R = set(get_entity_gp_re([dev_outputs[0][index], dev_outputs[1][index], dev_outputs[2][index]],
                                             dev_batch_data['attention_masks'][index].detach(),
                                             self.id2label))
                    T = set(dev_batch_data['callback'][index])
                    X_all += len(R & T)
                    Y_all += len(R)
                    Z_all += len(T)
                    for item in R & T:
                        X[item[2]] += 1
                    for item in R:
                        Y[item[2]] += 1
                    for item in T:
                        Z[item[2]] += 1

        len1 = max(max([len(i) for i in entitys]), 4)
        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all
        str_log = '\n{:<10}{:<15}{:<15}{:<15}\n'.format('关系' + chr(12288) * (len1 - len('关系')), 'precision', 'recall',
                                                        'f1-score')
        str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format('全部关系' + chr(12288) * (len1 - len('全部关系')), precision,
                                                                recall, f1)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        for entity in entitys:
            str_log += '{:<10}{:<15.4f}{:<15.4f}{:<15.4f}\n'.format(entity + chr(12288) * (len1 - len(entity)),
                                                                    precision[entitys_to_ids[entity]],
                                                                    recall[entitys_to_ids[entity]],
                                                                    f1[entitys_to_ids[entity]])
        self.log.info(str_log)


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """

    def __init__(self, mask_zero=False, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]
        return pos_loss + neg_loss


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class MyLossNer(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0]*y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0]*y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


class MyLossRe(SparseMultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_trues):
        ''' y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]
        '''
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # [btz, heads, 实体起终点的下标]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))  # [btz, heads, seq_len*seq_len]
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return sum(loss_list) / 3, loss_list[0], loss_list[1], loss_list[2]
