# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/2/16
@Time    : 16:10
@File    : torch_cache.py
@Function: XX
@Other: XX
"""

import sys
import torch


a = torch.zeros(60000, 63980, dtype=torch.float, device='cuda:0')
del a  # 只加这句并不能减少显存
torch.cuda.empty_cache()  # 成功减少了显存
# 在复现DQN的经验回访池时，发现我的小显卡训练2w个epoch就爆显存了，检查后才发现是忘了清torch的cache
sys.exit(0)
