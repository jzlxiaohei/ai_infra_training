# 大规模 GPU 训练基础设施 — 4 周学习计划

> 目标：对 JD 里的关键词（DDP/FSDP/DeepSpeed、K8s/Slurm、profiling、NCCL、千卡训练）建立**体感和词汇表**，能在面试里说清楚每个东西解决什么问题、瓶颈在哪。
>
> 不追求精通，追求"能聊得起来 + 跑通过 demo"。

---

## 0. 云 GPU 环境推荐（新加坡访问，海外平台）

### 主力：RunPod（推荐）
- 网址：runpod.io
- **有新加坡数据中心**（SG region），延迟低；US/EU 区域更便宜，SSH/训练对延迟不敏感也可以用
- 两种实例：
  - **Community Cloud**：便宜，可能被抢占，适合 W1-W2 学习
  - **Secure Cloud**：贵 ~50%，稳定，适合 W4 demo 长跑
- 价格参考（Community Cloud，USD/小时）：
  - 单卡 RTX 4090：~$0.34
  - **2× RTX 4090：~$0.70（W1-W2 主力配置）**
  - 单卡 A100 80G SXM：~$1.19
  - 4× A100 80G：~$4.76（W2 短租摸 FSDP + 大模型）
  - 8× H100：~$16-20（不需要，列出来知道大概水平）
- 优点：信用卡支付即可、UI 干净、有 PyTorch 官方模板镜像、Network Volume 可挂载持久化数据
- 计费方式：按秒计费，停掉实例 storage 仍按小时收（很便宜，~$0.10/月/GB），用完记得彻底删

### 备选
- **Vast.ai**：marketplace 模式，最便宜（4090 低至 $0.25/hr），但宿主机稳定性参差，适合 W1-W2 不在意中断的实验
- **Lambda Labs**：lambdalabs.com，A100/H100 质量好，$1.29/hr 起（但热门卡经常 sold out），W2 burst 抢到了就用
- **Paperspace (DigitalOcean Gradient)**：UI 友好，notebook 体验好，贵一些（A100 ~$3.18/hr），适合 W3-W4 不想折腾环境时
- **Modal**（modal.com）：serverless GPU，按秒计费、无需管实例。**W4 demo 想体验"提交训练任务"工作流时强推**，几行 Python 装饰器就能把函数跑到 A10/A100 上
- **Kaggle Notebook**：免费 2× T4 + 30小时/周，W1 sanity check 够用
- **Colab Pro+**：$50/月，有概率分到 A100，notebook 场景方便
- **GCP Singapore (asia-southeast1) / AWS ap-southeast-1**：本地区域，但 GPU quota 要申请、价格贵 2-3 倍，只在你想顺便摸大厂云时用

### 预算估计（USD）
全程 **$50-70（约 S$70-95）** 够用：
- W1-W2 主力 2×4090 Community，约 25-30 小时 × $0.70 ≈ $20
- W2 短租 4×A100，2-3 小时 × $4.76 ≈ $12
- W3-W4 demo（含 Secure Cloud 跑稳定些），约 15-20 小时 × $1-1.5 ≈ $20
- 持久存储 50GB × 1 个月 ≈ $5

比国内云便宜的原因：4090 在海外按 spot 价跑得很狠。

### 关键纪律
- **每次开机前列清单**：今天要跑什么，跑完立刻 stop（注意是 stop 不是 pause，pause 仍按 GPU 收费）
- **代码放 GitHub，不要只存实例**：community cloud 实例可能被回收
- **数据集用 HuggingFace `datasets` 库 + 缓存到 Network Volume**，避免每次重下
- **W2 跑 4×A100 之前先在 2×4090 把代码跑通**，A100 实例时间不要花在 debug Python 错误上
- **RunPod 注册时充 $10 起步即可**，不要一次性大额充值（防止账户问题）

---

## Week 1 — PyTorch 基础 + DDP

### 目标
- 看到一段训练代码能讲清楚 forward/backward/optimizer step 在做什么
- 跑通单机多卡 DDP，能解释 `all-reduce` 在 backward 里发生在哪一步

### 任务
1. **PyTorch 训练循环手写一遍**（CIFAR-10 + ResNet18，单卡）
   - 不用 Lightning/HuggingFace Trainer，原生 `for batch in loader` 写
   - 跑 1-2 个 epoch 能收敛即可
2. **改成 DDP**（2× 4090）
   - `torchrun --nproc_per_node=2 train.py`
   - 读懂 `DistributedSampler`、`DDP` 包裹模型、`dist.init_process_group`
3. **观察**：用 `nvidia-smi -l 1` 看两张卡是否同时打满

### 资料
- PyTorch 官方 DDP tutorial（一定看官方的，blog 经常过时）
- 《What Every User Should Know about Mixed Precision Training》PyTorch blog
- 视频：李沐《动手学深度学习》分布式训练那一节（B 站）

### Deliverable
- `week1/train_single.py` + `week1/train_ddp.py`
- 一段 200 字笔记：DDP 和 DataParallel 的区别、为什么前者快

---

## Week 2 — FSDP + ZeRO 概念（**重点**）

### 目标
- 能在白板上画出 ZeRO-1 / ZeRO-2 / ZeRO-3 各 shard 了什么
- 跑通 FSDP，看到显存占用变化

### 为什么选 FSDP 不选 DeepSpeed
原生在 PyTorch 里、API 更干净；ZeRO 概念是通用的，懂 FSDP 等于懂 DeepSpeed 的 80%。DeepSpeed 留着以后真用到再学。

### 任务
1. **读 ZeRO 论文的 Figure 1**（就那一张图，理解 optimizer states / gradients / parameters 三层 shard）
2. **改 W1 代码用 FSDP**，模型换成 GPT-2 small（~124M）
   - 对比 DDP vs FSDP 显存占用
3. **短租 4×A100**，跑一次 GPT-2 medium（~350M）+ FSDP
   - 体感：什么是 `auto_wrap_policy`、什么是 `cpu_offload`
4. **读一篇 blog 了解 3D 并行**（Megatron-LM 风格）
   - 关键词：Tensor Parallel（层内切）、Pipeline Parallel（层间切）、Data Parallel（batch 切）
   - **不用跑**，知道千卡训练靠的是 TP+PP+DP 组合即可

### 资料
- PyTorch FSDP tutorial（官方）
- ZeRO 论文 Section 3（其余可跳）
- HuggingFace 的《Efficient Training on Multiple GPUs》文档
- Blog：《How to Train Really Large Models》by Lilian Weng

### Deliverable
- `week2/train_fsdp.py`
- 笔记：DDP / FSDP / ZeRO-3 / TP / PP 一张对比表（显存占用、通信量、适用规模）

---

## Week 3 — K8s + GPU 调度（+ Slurm 半天）

### 目标
- 知道一个训练任务从"提交"到"在 GPU 上跑起来"中间发生了什么
- 能写 Pod yaml 申请 GPU

### 任务
1. **本地装 minikube 或 kind**，跑通基础 Pod/Deployment（不用 GPU）
2. **了解 GPU 调度链路**（读，不用本地复现完整链路）：
   - NVIDIA device plugin 怎么把 GPU 暴露给 K8s
   - `resources.limits.nvidia.com/gpu: 2` 是怎么生效的
   - Kubeflow Training Operator / Volcano 解决了什么问题（gang scheduling）
3. **在 AutoDL 实例上**用 Docker 跑训练镜像（不是 K8s，但理解容器化训练的工作流）
   - 写 Dockerfile，build，run with `--gpus all`
4. **Slurm 半天**：JD 列了 Slurm，HPC/研究环境用得比 K8s 多
   - 知道 `sbatch` / `srun --gres=gpu:2` / `squeue` 三个命令
   - 读一篇 blog 对比 Slurm vs K8s for ML（Anyscale 或 Lambda 的博客）
5. **Ray 半小时**：知道 Ray Train 是干嘛的就行，不用跑

### 资料
- 《Kubernetes in Action》前 4 章足够
- NVIDIA 的 K8s GPU Operator 文档（扫读）
- Kubeflow Training Operator README

### Deliverable
- `week3/Dockerfile` + `week3/train-job.yaml`（哪怕只在 minikube 验证调度，不真跑训练）
- 笔记：K8s / Slurm / Ray 各自适合什么场景

---

## Week 4 — Demo + Profiling/Monitoring（**面试关键**）

### 目标
做一个端到端的小型训练平台 demo，覆盖：**提交 → 调度 → 训练 → 监控 → 日志**

> Profiling/monitoring 单独是 JD 的一条，最容易被追问。务必动手摸过。

### 任务
1. **平台 demo**（在 AutoDL 单实例 + minikube，或直接 docker-compose）
   - 用户提交：一个 yaml 或 CLI 命令
   - 调度：K8s Job + GPU resource request
   - 训练：W2 的 FSDP 脚本
   - 日志：stdout 重定向到文件 + 简单 web 查看
2. **Profiling 至少摸过这三样**：
   - `torch.profiler`：导出 chrome trace，能看到 forward/backward/all-reduce 分别耗时
   - `nvidia-smi dmon` / `nvidia-smi pmon`：实时看 SM 占用率、显存带宽
   - `py-spy` 或 `torch.utils.bottleneck`：看 Python 端瓶颈
3. **Monitoring 搭一个最小栈**：
   - DCGM Exporter + Prometheus + Grafana（有现成 docker-compose 配置）
   - Grafana dashboard 至少看：GPU util、显存占用、SM 活跃度、温度
   - **关键指标理解**：MFU（Model FLOPs Utilization）是什么、为什么真实训练 MFU 经常只有 30-50%
4. **可选加分**：跑一次故意造瓶颈的实验
   - 故意把 batch_size 设小让 GPU 饿死，看 util 掉到多少
   - 故意关掉 mixed precision，看显存和速度变化

### 资料
- PyTorch Profiler 官方 tutorial
- 《Making Deep Learning Go Brrrr From First Principles》by Horace He（必读，面试金句来源）
- DCGM Exporter + Prometheus 的 GitHub README

### Deliverable
- `week4/demo/`：完整可运行的 demo（README 说明怎么跑）
- `week4/profiling-report.md`：贴 chrome trace 截图 + Grafana 截图，写 3 个发现的瓶颈
- **录一段 5 分钟自己讲解 demo 的视频**（面试前过一遍很有用）

---

## 必读补充清单（穿插在 4 周里，别专门留时间）

### 通信 / NCCL（W2-W3 期间读，1-2 小时）
JD 里 "communication bottlenecks" 的核心。你单机两卡感觉不到，但必须知道：
- All-reduce 的 ring 算法 vs tree 算法
- NVLink（机内）vs InfiniBand / RoCE（机间）带宽差几个数量级
- 为什么千卡训练时通信时间能占 30%+
- `NCCL_DEBUG=INFO` 这个环境变量

资料：NVIDIA 的 NCCL 官方文档"Overview"那一节、一篇关于 ring all-reduce 的 blog

### 千卡训练的真实样貌（W4 前读，1 小时）
- Meta 的 OPT-175B logbook（公开的，记录了他们训练时遇到的所有故障）
- Bloom 训练的工程 blog
- 知乎搜"千卡训练"，几篇国内大厂工程师写的实践文

读完你会对"为什么这个岗位值钱"有很真实的体感。

---

## 4 周后的自我检查（面试前问自己）

能用 2 分钟讲清楚每一个：

- [ ] DDP 和 FSDP 的区别，FSDP 省显存的代价是什么
- [ ] ZeRO-1/2/3 各 shard 了什么，通信量分别是 DDP 的多少倍
- [ ] 为什么千卡训练要用 3D 并行，单纯堆 DP 为什么不行
- [ ] K8s 调度 GPU 任务时，gang scheduling 解决了什么问题
- [ ] MFU 是什么，怎么从一次训练 run 算出来
- [ ] 一个训练 job 跑得慢，你会按什么顺序去 debug（GPU util? 通信? 数据加载? CPU?）
- [ ] NCCL all-reduce 在 DDP backward 的哪一步发生

任何一条说不清楚，回去补对应那周的内容。

---

## 时间投入

每周 **10-15 小时**，4 周共 40-60 小时。投入低于这个量，建议把范围砍到 W1-W2 + W4 的 profiling 部分，宁可窄但扎实。
