# Week 1 — PyTorch + DDP

## 目标
- 跑通单机单卡训练，能讲清训练循环 5 步
- 跑通单机 2 卡 DDP，看到两张卡同时打满
- 写一段 200 字笔记：DDP vs DataParallel

## 文件
- `train_single.py` — 单设备（CPU 或单卡 GPU）
- `train_ddp.py` — DDP 多卡版本，对照 `# DDP:` 注释看差异
- `notes.md` — 你的笔记（自己建）

---

## Step 0 — 本地 CPU smoke test（省钱，强烈建议先做）

在你的 Mac 上：
```bash
cd /Users/bytedance/personal/training/week1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_single.py --device cpu --epochs 1 --subset 500
```
跑通 = 代码无低级错误，可上 GPU。loss 不收敛没关系，subset 太小。

---

## Step 1 — RunPod 起 2× 4090

1. runpod.io 注册 → Billing 充 $10
2. **Pods → Deploy** → GPU type: `2× RTX 4090`，Cloud Type: `Community`
3. Template: **RunPod PyTorch 2.1**（已装 torch/torchvision）
4. Container Disk: 20GB；Volume: 留空
5. **Deploy On-Demand** → 等 30 秒 → **Connect** → Web Terminal 或 SSH

把代码弄上去（任选一种）：
- 推荐：`git init` 本地，push 到 GitHub 私库，pod 里 `git clone`
- 简单：RunPod 网页端 File Browser 直接上传 `week1/` 目录

---

## Step 2 — 单卡跑

```bash
cd /workspace/week1   # 或你放代码的路径
python train_single.py --epochs 2
```
预期：每 epoch 30-60 秒，2 epoch 后 `test_acc` ~50-65%。

**新开一个 terminal 同时跑**：
```bash
nvidia-smi -l 1
```
观察：只有 `GPU 0` 在跑，`GPU 1` Util=0%。

---

## Step 3 — DDP 跑

```bash
torchrun --nproc_per_node=2 train_ddp.py --epochs 2
```
预期：
- 两张卡 Util 都到 ~90%+
- 每 epoch 时间约为单卡的 60-70%（不是精确 50%，因为有通信开销）
- per-rank batch=128，所以 global batch=256，loss 数值和单卡不严格可比

---

## Step 4 — 实验观察清单

每跑一项在 `notes.md` 写 1-2 行：
- [ ] 单卡 `nvidia-smi` 看到的 Util、显存、功耗
- [ ] DDP 时两卡 Util、显存、温度
- [ ] DDP 比单卡 epoch 时间快多少倍？为什么不是精确 2 倍？
- [ ] 改 `--nproc_per_node=1` 后，DDP 还能跑吗？发生了什么？
- [ ] 注释掉 `train_sampler.set_epoch(epoch)` 那一行，跑两个 epoch，结果怎么变？为什么？
- [ ] 故意把 `--batch-size 1024` 设很大，OOM 在哪一步抛出？错误信息长什么样？

---

## Step 5 — 200 字笔记（写在 `notes.md`）

主题：**DDP vs DataParallel**。要点提示（命中 4-5 个就够）：
- DataParallel = 单进程多线程，主卡聚合 grad → 主卡瓶颈 + Python GIL
- DDP = 多进程，每卡一个进程，gradient 通过 NCCL all-reduce 同步
- DDP 的 backward 把 gradient 切成 buckets，**通信和计算 overlap**（一边算一边发）
- DDP 跨机也能用，DataParallel 只能单机
- PyTorch 官方文档明确说"用 DDP 不用 DataParallel"

---

## 自检（W1 结业，不看代码能讲清楚）

- [ ] 训练循环的 5 行核心是哪 5 行，各做什么
- [ ] DDP 里 `loss.backward()` 这一行除了算梯度还做了什么
- [ ] 为什么 DDP 要用 `DistributedSampler`，不用会怎样
- [ ] `torchrun --nproc_per_node=2` 启动时，每个子进程的 `RANK` / `LOCAL_RANK` / `WORLD_SIZE` 是什么
- [ ] `dist.all_reduce(SUM)` 和 `dist.all_gather` 区别

---

## 关机！

跑完每一段立刻：
- RunPod 点 **Stop** （保留磁盘按 storage 收费，便宜）
- W1 全部结束后建议 **Terminate**（W2 重新建实例，避免忘记关）
- 再次提醒：**Pause ≠ Stop**，Pause 仍然按 GPU 收钱
