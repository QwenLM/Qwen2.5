# Qwen2.5 推理

## 1、配置运行环境

**安装驱动**

```
# <version_id> 为软件包具体版本号。
chmod +x TopsRider_i3x_<version_id>_deb_amd64.run
./TopsRider_i3x_<version_id>_deb_amd64.run -y
```

**创建并启动 docker**

```
# 创建 docker 容器，将在基础镜像 artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_2004_gcc9:1.4.4 的基础上创建 docker。
# <project_path> 当前工程所在路径
# -e ENFLAME_VISIBLE_DEVICES=2 进行 GCU 资源隔离，如需多卡可以改为 0,1,2,3 等
docker run -itd -e ENFLAME_VISIBLE_DEVICES=2 --name qwen-infer -v <project_path>:/work -v /root/:/root/ --privileged --network host  artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_2004_gcc9:1.4.4 bash
```

**进入 docker 安装环境**

```
# 进入 docker 容器
docker exec -it qwen-infer bash

# 安装 SDK 框架，进入软件包所在地址。
# <version_id> 为软件包具体版本号。
./TopsRider_i3x_<version_id>_amd64.run -C torch-gcu-2 -y
./TopsRider_i3x_<version_id>_deb_amd64.run -C tops-sdk -y

# 安装 python 库
pip3.8 install transformers==4.40.2
pip3.8 install accelerate
```

## 2、推理

```
# 进入本工程目录，包含运行代码、推理输入等文件。
.
├── README.md
└── gcu_demo.py
```

**启动推理示例**

```
python3.8 gcu_demo.py
```
执行 gcu_demo.py 推理示例，代码改编自 [仓库 README](https://github.com/QwenLM/Qwen2.5/blob/main/README.md) 中的给的 Huggingface quick start 用例。

**GCU PyTorch 原生推理支持**

GCU 支持 pytorch 原生推理，在 pytorch 代码上只需做少许改动就可以在 GCU 上顺利运行：

1. 导入 *torch_gcu* 后端库，并载入 transfer_to_gcu
    ``` python
    try:
        import torch_gcu # 导入 torch_gcu
        from torch_gcu import transfer_to_gcu #  transfer_to_gcu
    except Exception as e:
        print(e)
    ```
2. device 名改为 *gcu*
   ``` python
   device = "gcu"
   ```

**GCU vLLM 推理**

GCU 也支持 *vLLM* 原生推理，需要安装 GCU 版本的 *vLLM* 后，将设备名改为 gcu

```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2.5-7B-Instruct --model Qwen/Qwen2.5-7B-Instruct --device gcu
```
