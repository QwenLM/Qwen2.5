# Qwen1.5 推理

## 1、配置运行环境

**安装驱动**

```
# <version_id> 为软件包具体版本号。
chmod +x TopsRider_i2x_<version_id>_deb_internal_amd64.run
./TopsRider_i2x_<version_id>_deb_internal_amd64.run -y
```

**创建并启动 docker**

```
# 创建 docker 容器，将在基础镜像 artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_2004_gcc9:1.4.4 的基础上创建 docker。
# <project_path> 当前工程所在路径
docker run -itd --name qwen-infer -v <project_path>:/work -v /root/:/root/ --privileged --network host  --ipc host artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_2004_gcc9:1.4.4 bash
```

**进入 docker 安装环境**

```
# 进入 docker 容器
docker exec -it qwen-infer bash

# 安装 SDK 框架，进入软件包所在地址。
# <version_id> 为软件包具体版本号。
./TopsRider_i2x_<version_id>_deb_internal_amd64.run -y -C torch-gcu --python python3.8
./TopsRider_i2x_<version_id>_deb_internal_amd64.run -y --python python3.8
./TopsRider_i2x_<version_id>_deb_internal_amd64.run -y -C topstransformer

# 安装 python 库
pip3.8 install transformers==4.37.0
```

## 2、推理

```
# 进入本工程目录，包含运行代码、推理输入等文件。
.
├── inferQwen.py
├── qwen1.5.ini
├── README.md
├── test.txt
└── weight_preprocess_qwen1.5_chat.py
```

**预训练模型地址**

预训练模型采用 [Qwen1.5-14B-Chat](https://www.modelscope.cn/models/qwen/Qwen1.5-14B-Chat/files) ，可自行下载模型文件，保存到本工程目录下。

**拆分预训练模型**

```
python3.8 weight_preprocess_qwen1.5_chat.py -tp 2 \
                                            -i ./Qwen1.5-14B-Chat \
                                            -o ./Qwen1.5-Split-TP2 \
                                            -se True
```
-tp 设置张量并行尺寸，支持 2 ( seq_length 最大可支持到 4K )或 4 ( seq_length 最大可支持到 8K+ )，-i 设置预训练模型的地址，-o 设置切分后模型地址，-se 设置是否切分 embedding 层，执行完该步骤后会对原始的预训练模型进行拆分。主要目的是将“大”模型拆分成多个“小”模型，在进行推理过程中，采用模型并行策略，提升模型推理效率，减小显存压力。

**启动推理**

```
python3.8 inferQwen.py -a ./Qwen1.5-14B-Chat \
                          -w ./Qwen1.5-Split-TP2 \
                          -t ./test.txt \
                          -b 1 \
                          -i ./qwen1.5.ini \
                          -tp 2
```
执行 inferQwen.py 推理脚本，其中 -a 设置读取 token_config_path 的地址,默认在原始预训练模型的路径下，-w 设置拆分后的模型地址，-i 设置模型的配置文件地址，-t 设置推理的输入文件地址，-tp 配置张量并行参数，输入文件中的格式如下：
```
今天天气怎么样？
作为一名教师，我该如何授课？
```