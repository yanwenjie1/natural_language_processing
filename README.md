# natural_language_processing
自然语言处理 pytorch实现

## 使用依赖

```python
python==3.10.0
torch==1.12.1
transformers==4.26.0
```

## 项目结构

```
--checkpoints：存放保存的模型，配置及训练日志
--data：数据集
--utils：工具箱
--config.py：配置信息
--train.py：训练
--server.py：模型服务接口
--server_confindence.py：模型服务接口 附带置信度
--test.py：测试
```

## 环境配置
- step1 基础环境确认
  - win+R 运行cmd 进入命令行界面
  - 确认机器有nvidia显卡
  - 输入 nvidia-smi 查看显卡驱动是否支持11.3版本cuda （保证CUDA Version大于11.3即可）
  - win+R 运行cmd 进入命令行界面 输入conda info -e 确认anaconda/miniconda安装成功
  - 建议配置conda源 参考 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
  - 建议配置pip源   参考 https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
- step2 构建模型环境
  - 运行 conda create -n yourname python=3.10.0
  - conda activate yourname / source activate yourname
  - 执行下面的命令安装torch环境 参考 https://pytorch.org/get-started/previous-versions/
  - conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  - 如果是cpu机器：conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
  - 执行下面的命令安装机器学习库scikit-learn==1.1.3
  - conda install scikit-learn  
  - conda install transformers==4.26.0
  - pip install pytorch-crf==0.7.2
  - pip install pynvml==11.4.1
  - pip install flask 
  - pip install gevent
 # 备注
 - 已测试支持的cuda版本：11.0-11.7
 - 已测试支持的pytorch版本：1.7.0-1.13.0
 - 已测试支持的预训练模型：
    - bert
    - albert
    - electra
    - roberta
    - gpt2
    - roformer
    - roformerV2
    - nezha
 - 推荐的预训练模型
    - chinese-albert-base-cluecorpussmall
    - chinese-bert-base
    - chinese-bert-wwm-ext
 - 预训练模型路径
    - \技术中心\0AI\DeepLearning\预训练模型
 - 语料路径
    - \技术中心\0AI\DeepLearning\通用语料 其下内容拷贝到data文件夹即可
    
 # 参考资料
 - https://github.com/taishan1994
 - https://huggingface.co/docs/transformers

 # 更新日志
 - 2022-11-29：优化了服务端，允许接受batch输入
 - 2022-12-05：优化了batch推理 降低了显存占用问题
 - 2023-01-28：从文本分类到事件抽取全部集成完毕
 - 2023-06-06：更新了NER-GP的server，完善了全部模型的置信度支持
 - 2023-06-20：增加了生成式模型的支持，优化了GP4NER的解码方式，完善了tqdm在linux下的表现
 - 2023-06-29：1.增加了对longformer的支持 2.从标注平台到模型推理 在实体识别任务和关系抽取任务复现了UIE模型 详情请参考CemeteryFundUIE项目
 - 2023-07-04：
     - 增加了对ernie的支持
     - 增加了对uie版本的ernie模型支持，详情参考CemeteryFundErnie项目
