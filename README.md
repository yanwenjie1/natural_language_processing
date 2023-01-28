# natural_language_processing
自然语言处理 Torch实现

## 使用依赖
```python
python==3.10.0
torch==1.12.1
transformers==4.26.0
```

## 项目结构
- **data**
  - **数据集**
    - ***preprocess.py*** 构造输入
    - ***train_data.txt*** 样本示例
- **utils**
  - ***utils.py*** 工具箱
  - ***bert_base_model.py*** bert类预训练模型加载
  - ***bert_ner_model.py*** seq2seq基本模型
  - ***bert_classify_model.py*** seq2one基本模型
  - ***adversarial_training.py*** 对抗学习实现
  - ***adversarial_example.ipynb*** 对抗学习示例
- **config.py** 配置文件
- **train.py** 训练文件
- **server.py** 接口文件
- **test.py** 测试文件
# 环境配置
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
  - 执行下面的命令安装机器学习库scikit-learn==1.1.3
  - conda install scikit-learn  
  - conda install transformers==4.18.0
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
 - 推荐的预训练模型
    - chinese-albert-base-cluecorpussmall
    - chinese-bert-base
    - chinese-bert-wwm-ext
 - 预训练模型路径
    - \技术中心\0AI\DeepLearning\预训练模型
    
 # 参考资料
 - https://github.com/taishan1994
 - https://huggingface.co/docs/transformers

 # 更新日志
 - 2022-11-29：优化了服务端，允许接受batch输入
 - 2022-12-05：优化了batch推理 降低了显存占用问题
