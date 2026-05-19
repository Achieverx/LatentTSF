# Decoder-Compatible Future-Separable Latent Dynamics — V1

## 方法定位

LatentTSF 的 decoder-friendly latent space 不应被重训破坏。
FA-EMA 已证明 latent MSE 下降不等于 observation MSE 下降（decoder-compatibility gap）。
因此本方法：完全冻结原 LatentTSF 的 Encoder/Decoder/BaseForecaster，
在 decoder-compatible manifold 附近学习多个 input-conditioned future latent candidates，
通过 observation-space softmin 监督 + adaptive fusion 改善最终预测。

脚本名：train_latent_candidates.py

---

## 前提验证（必须先跑，再训练）

在正式训练前，用 frozen encoder/decoder 在 z_base 附近做随机扰动搜索：

```python
for _ in range(200):
    delta  = torch.randn_like(z_base)           # 不要乘 beta，下面 tanh 之后才乘
    z_cand = z_base + beta * torch.tanh(delta)  # beta 只乘一次
    y_cand = Decoder_old(z_cand)
    track min MSE(y_cand, y)
```

输出 oracle_obs_gain = base_obs_mse - best_random_obs_mse。
如果 oracle_obs_gain <= 0（在 validation set 上），打印强警告。
可通过 --force_train 1 强制继续训练，但必须在日志和 metrics json 里记录该警告。

---

## 核心流程

```python
# 全部 frozen，只提供表示
z_x    = Encoder_old(batch_x).detach()       # [B, seq_len, D]
z_y    = Encoder_old(batch_y).detach()       # [B, pred_len, D]
z_base = BaseForecaster_old(z_x).detach()    # [B, pred_len, D]
y_base = Decoder_old(z_base).detach()        # [B, pred_len, C]

# 生成 M 个 decoder-compatible candidates
delta  = CandidateGenerator(z_x, z_base)     # [B, M, pred_len, D]
z_cands = z_base[:,None] + beta * tanh(delta) # [B, M, pred_len, D]
y_cands = Decoder_old(z_cands)               # [B, M, pred_len, C]

# Fusion
w      = softmax(FusionGate(z_x, z_base, z_cands), dim=1)  # [B, M]
z_fused = z_base + sum_m w_m * (z_cands_m - z_base)
y_hat   = Decoder_old(z_fused)               # [B, pred_len, C]
```

默认 M=4，beta=0.2。
CandidateGenerator 最后一层 zero-init，使初始 z_cands ≈ z_base。

---

## Loss

```
L = lambda_forecast * L_forecast
  + lambda_set      * L_set
  + lambda_div      * L_div
  + lambda_anchor   * L_anchor
```

**L_forecast** = MSE(y_hat, batch_y)

**L_set（混合 softmin，防止 mode collapse）**：
```python
dist_obs_m = MSE(y_cands_m, batch_y)        # [B, M]
dist_lat_m = MSE(z_cands_m, z_y)            # [B, M]

# Loss 项：用 softmin（可微）
softmin_obs_loss = (-tau * logsumexp(-dist_obs_m / tau, dim=1)).mean()
softmin_lat_loss = (-tau * logsumexp(-dist_lat_m / tau, dim=1)).mean()
L_set = softmin_obs_loss + 0.5 * softmin_lat_loss

# Diagnostic 项：用真实 min（不可微，只用于报告）
min_obs_mse    = dist_obs_m.min(dim=1).values.mean()   # 真实 oracle 上界
min_latent_mse = dist_lat_m.min(dim=1).values.mean()
```
softmin 只用于 loss 计算；oracle 报告必须用 true min，否则判断失真。
tau=0.5（不要用 0.1，太尖锐会导致 candidate mode collapse）。

**L_div（防止所有 candidate 退化为同一个）**：
```python
# pairwise L2 distance between candidates
z_flat = z_cands.mean(dim=2)               # [B, M, D]
pairwise = pdist(z_flat)
L_div = -pairwise.mean()
```
lambda_div=0.005（不要太小，0.001 不够）。

**L_anchor（防止 candidate 漂出 decoder manifold）**：
```python
L_anchor = (z_cands - z_base[:,None]).pow(2).mean()
```
lambda_anchor=0.05。

默认权重：
```
lambda_forecast = 1.0
lambda_set      = 0.3
lambda_div      = 0.005
lambda_anchor   = 0.05
tau             = 0.5
```

---

## CandidateGenerator 结构

```python
# 输入: z_base [B,H,D], context = z_x.mean(1) [B,D]
ctx = context[:,None,:].expand(-1, H, -1)   # [B,H,D]
inp = cat([z_base, ctx], dim=-1)            # [B,H,2D]

# temporal conv
x = inp.permute(0,2,1)                      # [B,2D,H]
x = Conv1d(2D, 4D, k=3, p=1) -> GELU -> Dropout
x = Conv1d(4D, 4D, k=3, p=1) -> GELU -> Dropout
x = Conv1d(4D, M*D, k=1)                   # [B,M*D,H]
x = x.permute(0,2,1).reshape(B,H,M,D).permute(0,2,1,3)  # [B,M,H,D]
delta = x  # 最后一层 zero-init
```

不要用 z_x last token，只用 mean，更稳。

---

## FusionGate 结构

```python
# 每个 candidate 的 summary
cand_mean = z_cands.mean(dim=2)             # [B,M,D]
cand_res  = (z_cands - z_base[:,None]).mean(dim=2)  # [B,M,D]
base_ctx  = cat([z_base.mean(1), z_x.mean(1)], dim=-1)  # [B,2D]
base_ctx  = base_ctx[:,None,:].expand(-1,M,-1)           # [B,M,2D]

gate_inp  = cat([cand_mean, cand_res, base_ctx], dim=-1) # [B,M,4D]
score     = MLP(gate_inp).squeeze(-1)                    # [B,M]
w         = softmax(score, dim=1)
```

---

## Diagnostics（每 epoch train/val/test）

```
# Forecast
base_obs_mse          # MSE(y_base, y)，frozen baseline
base_latent_mse       # MSE(z_base, z_y)
fused_obs_mse         # MSE(y_hat, y)，最终输出
fused_obs_gain        # base_obs_mse - fused_obs_mse

# Oracle（true min，不是 softmin）
min_obs_mse           # min_m MSE(y_cands_m, y)，方法上界
min_latent_mse        # min_m MSE(z_cands_m, z_y)
oracle_obs_gain       # base_obs_mse - min_obs_mse

# Loss components（分开记录，方便 debug）
softmin_obs_loss
softmin_lat_loss
loss_div
loss_anchor
loss_total

# Candidate behavior
delta_abs_mean        # mean |delta| before tanh
delta_abs_max         # max |delta|
candidate_std_across_M # std of z_cands over M dim，监控 diversity
pairwise_diversity    # pairwise L2 between candidates
anchor_dist           # mean ||z_cands - z_base||

# Fusion
weight_entropy        # -sum w log w
max_weight_mean       # mean of max(w)，监控单 candidate 主导

# Block-wise（pred_len 切 K=4 blocks）
block_obs_mse_b0..bK  # fused output 每段误差
block_base_obs_b0..bK # base 每段误差对比
```

关键判断逻辑：
- oracle_obs_gain ≈ 0 → 前提不成立，停止（或 --force_train 1 继续但记录警告）
- oracle 好但 fused 差 → FusionGate 是瓶颈
- fused 好 → 方法成立
- candidate_std_across_M 很小 → diversity collapse，检查 lambda_div 和 tau

---

## Ablation

```
--M              1 / 4 / 8
--beta           0.05 / 0.1 / 0.2 / 0.5
--tau            0.1 / 0.5 / 1.0
--fusion_mode    uniform / learned（默认）
--no_obs_set     lambda_set 去掉 obs 项
--no_latent_set  lambda_set 去掉 lat 项
--no_div
--no_anchor
```

---

## 第一轮实验

数据：ETTh1，seq_len=96，pred_len=96
Base model：DLinear LatentTSF checkpoint

顺序：
1. 先跑前提验证脚本，确认 oracle_obs_gain > 0
2. 再跑完整训练

```bash
python -u train_latent_candidates.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 --pred_len 96 \
  --base_checkpoint ./checkpoints/LatentTSF_DLinear_ETTh1_pl96/checkpoint.pth \
  --M 4 --beta 0.2 --tau 0.5 \
  --lambda_set 0.3 --lambda_div 0.005 --lambda_anchor 0.05 \
  --epochs 20 --patience 5 \
  --lr 3e-4 --batch_size 32 \
  --output_dir ./checkpoints/latent_cands_ETTh1_pl96
```

---

## 实现注意事项

1. **Frozen 模块必须不进 optimizer**：
   ```python
   for m in [Encoder_old, Decoder_old, BaseForecaster_old]:
       for p in m.parameters():
           p.requires_grad = False
   optimizer = AdamW([*CandidateGenerator.parameters(),
                      *FusionGate.parameters()], lr=lr)
   ```
   启动时打印：trainable params / frozen params，确认没有漏 frozen。

2. **beta 只乘一次**：`z_cands = z_base[:,None] + beta * tanh(delta)`，
   delta 是 CandidateGenerator 的原始输出，不要在生成时再乘 beta。

3. **oracle 用 true min，loss 用 softmin**：两者不要混用，否则 diagnostic 判断失真。

4. **CandidateGenerator 最后一层必须 zero-init**，保证初始 z_cands ≈ z_base。

5. **tau 不要用 0.1**，会导致只有一个 candidate 有梯度，其余退化。

6. **前提验证 oracle_obs_gain <= 0 时**：打印 WARNING，记录到 metrics json，
   默认停止训练；--force_train 1 可强制继续。

7. **输出文件**：
   ```
   output_dir/
   ├── best_checkpoint.pth
   ├── epoch_metrics.csv       # 每 epoch 所有 diagnostic 指标
   ├── test_diagnostics.json   # test set oracle/fusion/block 完整指标
   └── metrics_summary.json    # 最终 best val 指标汇总 + premise check 结果
   ```

8. **pairwise diversity** 用 `z_cands.mean(dim=2)` 而不是全序列，节省内存。
