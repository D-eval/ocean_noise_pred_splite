# Forecast-conditioned ocean source separation

当前论文主方法以过去 60 秒事件掩蔽海洋声景预测下一段 `79×128` 背景图，并将其与显式创新图送入残差适配器及 BS-RoFormer。下列旧频谱条件、FiLM、拼接和交叉注意力实验保留为前期消融：

1. `none`：不做噪声预测，直接训练条件置零的 BS-RoFormer；
2. `latent`：将 GRU 历史编码器的预测潜表示注入 band-split token；
3. `spectrum`：将预测的当前 TOL 频谱注入同一位置。
4. `concat`：逐子带投影后与音频 token 拼接，再由 MLP 融合；
5. `film`：预测逐子带缩放与偏置；
6. `cross_attention`：音频 token 主动关注预测频率箱 token。

## 数据口径

默认使用 `third_party/soundscape/datasets` 中 SanctSound MB03 的原始 TOL CSV：MB03_01/02 训练、MB03_03 验证、MB03_04 测试。条件历史截止到 `t-1`，`t` 时刻频谱只作为预测监督和当前背景的构造依据，因此预测器看不到未来。

公开的 TOL 产品没有原始波形相位。第一层实验中的“当前海洋噪声”是按当小时实测 TOL 包络生成的确定性随机相位代理波形，不是恢复出来的现场录音；它只用于验证联合训练和注入机制。第二层已用时间对齐的连续原始水听器背景复验，第三层进一步测试真实标注事件，但真实事件仍没有隔离声轨，因此不能据此声称完整波形保真或现场部署就绪。

## 环境与测试

项目会直接引用 `third_party/SpliteAnySound/third_party/BS-RoFormer`，无需复制其代码。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

快速端到端消融（小模型、短片段）：

```powershell
python -m source.run_ablation --small --steps 20 --batch-size 4 `
  --seconds 0.25 --validation-examples 16 --test-examples 32 `
  --train-examples-per-deployment 128 --validate-every 10
```

正式配置：

```powershell
python -m source.run_ablation --steps 1000 --batch-size 8 `
  --seconds 0.5 --validation-examples 128 --test-examples 256 `
  --validate-every 100
```

当前部署模型是只按验证集选择的 FiLM：25.5 万参数、2000 次更新、六类广分布信号和 −12 至 +12 dB 训练范围。仓库包含约 1 MB 的检查点 `models/practical_film.pt`：

```powershell
python -m source.web
```

打开 `http://127.0.0.1:8000`，上传当前 WAV，并提供 60 秒（12 个连续 5-s 窗）的干净历史背景 WAV，或至少 12 行 event-masked 历史特征 CSV（每行一个 5-s 窗），即可试听、下载目标信号与海洋背景。浏览器数据只发送到本机后端。端到端 RTF 取决于当前训练模型与硬件。

输出保存在 `experiments/ablation/<variant>/`。模型权重被 `.gitignore` 排除；小型 JSON/CSV 指标可选择性纳入版本控制。总表 `ablation_summary.json` 给出两种条件相对无条件基线的配对 SI-SDRi 增益、bootstrap 95% 区间和逐样本胜率。

## 方法摘要

论文按三层证据组织，避免把“真实背景”和“真实目标”混为一谈：第一层用 TOL 构造的代理背景与合成目标完成注入机制消融；第二层换成连续真实海洋背景，但保留合成目标以获得完整分离真值；第三层再用真实标注事件测试语义迁移，此时不再声称拥有分离真值。

历史频谱经 GRU 编码为潜变量并解码为下一时刻频谱。代理背景消融的联合目标为

`L = L_wave+MRSTFT + 0.2 * L_forecast`。

条件在 band split 后、轴向 Transformer 前注入。拼接方案用 MLP 压缩 `[音频;条件]`，FiLM 用条件控制每个子带的缩放和偏置，交叉注意力则以音频为 query、预测频率箱为 key/value；三者残差支路均从恒等映射初始化。第一层允许分离损失反向更新预测器；两项真实波形实验先按验证预测误差选出预测器并冻结，再训练分离器，以免分离损失暗中破坏预测准确率。详细定义、限制和正式实验协议见 [paper/main.tex](paper/main.tex)。

## 已完成的 smoke 结果

300 步小模型实验中，无条件基线、潜表示条件、频谱条件的测试 SI-SDRi 分别为 3.75、3.44、4.15 dB。潜表示相对基线下降 0.31 dB（配对 95% CI：−0.43 至 −0.20），预测频谱提高 0.39 dB（0.17 至 0.61）；三种方案均能快于实时运行。这个结果只支持 TOL 代理波形上的“显式预测频谱”接口，不能外推为现场原始音频结论。完整未四舍五入结果见 [results/smoke_ablation.json](results/smoke_ablation.json)。

## 旧版实用基线

1500 步实用模型总体 SI-SDRi 为 5.74 dB（95% CI：5.04–6.47），通过预注册工程门槛。−12/−6/0/+6/+12 dB 五档输入的 SI-SDRi 为 6.55/6.32/5.73/3.74/−0.14 dB；+12 dB 档的绝对目标 SI-SDR 为 11.89 dB，说明高强度输入已接近“无需处理”的区域。

同规模、同训练量的无条件 BS-RoFormer 基线为 6.02 dB；旧加法频谱条件相对基线的配对差为 −0.28 dB（95% CI：−0.69 至 0.14）。该负结果推动了后续广分布条件注入消融；历史结果保留在 [results/practical_results.json](results/practical_results.json)，不再作为部署模型。

论文提供英文 [main.tex](paper/main.tex) 与中文 [main_zh.tex](paper/main_zh.tex) 两份源文件，以及已渲染的[英文 PDF](output/pdf/forecast_conditioned_ocean_separation_en.pdf)和[中文 PDF](output/pdf/forecast_conditioned_ocean_separation_zh.pdf)。

## 广分布信号与散度实验

广分布生成器覆盖谐波、扫频、非谐波、脉冲列、调频和带限噪声六类信号，并放宽频率、包络、调制、瞬态与噪声占比范围。训练时加入 `--source-profile broad`：

```powershell
python -m source.train --variant none --output experiments/broad_none `
  --practical --source-profile broad --steps 2000
python -m source.train --variant concat --output experiments/broad_concat `
  --practical --source-profile broad --steps 2000
python -m source.train --variant film --output experiments/broad_film `
  --practical --source-profile broad --steps 2000
python -m source.train --variant cross_attention --output experiments/broad_cross_attention `
  --practical --source-profile broad --steps 2000
python -m source.diversity_eval `
  --baseline-checkpoint experiments/broad_none/best.pt `
  --conditioned-checkpoint experiments/broad_film/best.pt `
  --output results/diversity_film.json
python -m source.conditioning_ablation
```

三种方案共享数据、种子、分离器核心初始权重、训练步数和检查点规则，并只按 MB03_03 验证 SI-SDRi 选型：拼接、FiLM、交叉注意力分别为 `5.17/5.35/4.99 dB`，因此选择 FiLM。普通 MB03_04 测试分别为 `5.30/5.39/5.11 dB`，无条件基线为 `5.00 dB`。

固定 0 dB SNR、六类各 128 个完全配对样本上，三者相对无条件基线的增益分别为：拼接 `+0.148 dB`（95% CI `0.024–0.273`）、FiLM `+0.342 dB`（`0.207–0.478`）、交叉注意力 `+0.010 dB`（`−0.081–0.103`）。FiLM 用浅层逐带乘加直接表达“预期背景应抑制/平移哪些特征”，无需压缩加倍特征，也无需从粗粒度小时频谱学习复杂对齐，因此在当前数据规模下最好；这是机制解释，不是因果证明。

主分布偏移指标是每个源功率谱相对原始窄分布参考的 Jensen–Shannon 散度；同时报告方向性 KL 散度、源与海洋背景的频谱重叠，以及归一化频谱熵。FiLM 增益与 JS 的相关系数仅 `−0.085`，所以“偏离旧分布越远，先验越有用”不成立；最高熵五分位提升 `0.80 dB`（`0.47–1.17`），带限噪声、脉冲列和调频信号分别提升 `0.54/0.47/0.43 dB`。消融总表见 [results/conditioning_ablation.json](results/conditioning_ablation.json)，完整分层结果见 [results/diversity_results.json](results/diversity_results.json)，逐样本数据见 [results/diversity_results.csv](results/diversity_results.csv)。

论文图可通过以下命令从实验结果与检查点重新生成：

```powershell
python -m source.make_paper_figures
```

## 子小时真实波形实验

新的时间序列实验直接读取本机已有的 SanctSound MB03 连续录音，不再按 TOL 包络随机生成海洋背景。对 `5 s / 30 s / 60 s / 600 s` 四种 `window_len`，模型使用严格位于当前窗之前的 1 小时历史预测下一窗；随后把广分布合成信号加入该下一窗真实 WAV，并以原始 WAV 和合成信号作为干净监督。训练、验证和测试分别使用 MB03_01+02、MB03_03、MB03_04。

```powershell
python -m source.window_resolution_experiment --window 5
python -m source.window_resolution_experiment --window 30
python -m source.window_resolution_experiment --window 60
python -m source.window_resolution_experiment --window 600
```

正式配置为：预测器至少 1500、最多 2500 次更新，分离器至少 3000、最多 5000 次更新；选择规则只使用验证集，并把零步的“上一窗持续性预测”作为必须击败的候选。分离训练覆盖 8192 个混合片段，测试使用 1024 个配对片段，另在 `−12/−6/0/+6/+12 dB` 各评估 256 个片段。

四种分辨率相对同一真实波形无条件基线的 SI-SDRi 配对增益分别为 `+1.226 / +1.095 / +1.128 / +1.176 dB`，95% bootstrap 区间均高于零；5 秒配置的绝对 SI-SDRi 最高，为 `29.753 dB`。只有 5 秒预测器在验证集上略优于持续性预测（MAE 改善 `1.39%`），但该预测优势没有迁移到 MB03_04；30 秒、1 分钟和 10 分钟均由验证规则选择了持续性预测。因而当前结果证明短时、时间相关的背景条件有助于分离，但尚不能声称学习到的跨部署预测器优于简单持续性先验。

完整数字见 [results/window_resolution_results.json](results/window_resolution_results.json)，可缩放的矢量汇总图见 [paper/figures/window_resolution.pdf](paper/figures/window_resolution.pdf)，中英文论文已同步更新。

## 早期事件数据实验（已被下方主设置与训练命令替代）

DCLDE 2013 连续录音保存在 `data/dclde2013/`（约 2.3 GiB，已由 `.gitignore` 排除）。实验按录音日划分数据，用确定的露脊鲸上扫声作正样本；“干净背景”会排除确定及不确定事件前后 2 秒。训练阶段只把合成上扫声加入干净真实海洋片段，真实事件只用于最终无参考评估。

```powershell
python -m source.event_classifier --prepare-only
python -m source.event_classifier --epochs 30 --batch-size 64 --device cuda
```

冻结分类器在按日留出的测试集上达到 AUROC `0.9704`、AP `0.9738`、F1 `0.9196`。严格因果条件由当前片段之前的八个 8 秒窗构成；预测器的验证 MAE 为 `1.921 dB`，优于一步持续性基线的 `2.081 dB`（改善 `7.67%`）。旧零条件 FiLM 与 FC-BSR 使用相同真实干净背景、合成上扫声、随机种子、分离器初值、训练步数和验证选模规则，验证 SI-SDRi 分别为 `13.871` 和 `13.925 dB`。旧分离器命令已停用，复现当前主实验应使用下方命令。

## 前期主数据域实验：30 频带 FiLM 审计

前期设置固定为 **5 s + 12 个事件掩蔽历史 5-s 特征 + GRU + FiLM + BS-RoFormer**。每个样本是 `(H_t, N_t, s_t, x_t)`：`H_t` 是当前窗前连续 60 s 的 5-s 特征序列，历史中排除已标注事件及两侧 2 s；`N_t` 是真实、标注无事件的当前 5 s 片段；`s_t` 从 5 类鲸声代理与 2 类船噪代理均匀抽样（广分布生成器总共 13 类）；`x_t=N_t+s_t`。分离器只接收 `x_t,H_t`，以 `s_t` 为监督目标，背景输出由混合一致性残差得到。按录音日隔离后，146.4 h 原始干净当前窗经过连续历史筛选，94.5 h 可用于主实验，其中训练/验证/测试为 37.3/9.7/47.6 h。

重建事件掩蔽历史特征并训练预测器、零条件 FiLM 对照和主 FC-BSR：

```powershell
python -m source.dclde_separation_experiment --forecast-steps 5000 --separation-steps 10000 --batch-size 4 --device cuda
```

随后运行主数据域的六路 Oracle 实验（已有的零条件分离器与 GRU-Forecast 检查点直接复用；零条件模型会附上同一冻结 GRU，并先验证波形输出逐样本相同；其余四路按同一预算训练）：

```powershell
python -m source.dclde_oracle_experiment --device cuda
```

六路分别为真正跳过条件注入的 BS-RoFormer、零条件 FiLM、错时频谱 FiLM、最后历史频谱的 persistence FiLM、GRU-Forecast FiLM、当前真实干净背景频谱的 **30 频带 Oracle-FiLM**。后者使用当前样本的 `E(N_t)`，只检验现有压缩表示与 FiLM 接口，不能充当完整真实背景的上限。汇总、模型表和 4096 行配对逐例数据分别写到 `results/dclde_main_oracle.json`、同名 CSV 和 `results/dclde_main_oracle_samples.csv`；汇总脚本会逐例拒绝 ID、SNR 或混合基准分数不一致的模型输出。强度曲线由下方制图命令重建。

| 主域方法 | 留出目标 SI-SDRi (dB) | 相对真正 BSR 的配对差及日期分块 95% 区间 (dB) |
| --- | ---: | ---: |
| BS-RoFormer | 9.631 | 0 |
| Zero-condition FiLM | 9.631 | −0.0002 [−0.007, 0.009] |
| Shuffled-condition FiLM | 9.526 | −0.105 [−0.247, 0.020] |
| Persistence-condition FiLM | 9.524 | −0.107 [−0.239, 0.083] |
| GRU-Forecast FiLM | 9.497 | −0.134 [−0.327, 0.125] |
| 30 频带 Oracle-FiLM（当前真实 `E(N_t)`） | 9.535 | −0.096 [−0.238, 0.034] |

真正 BSR 与 Zero-FiLM 的总体差几乎为零；Oracle 的验证 SI-SDRi 高于 BSR（11.372 对 10.946 dB），但三个留出录音日上没有形成明确的正收益。Oracle−Forecast 为 +0.039 dB，日期分块区间 [−0.090, 0.155]；因此不能把预测误差单独认定为瓶颈。该 Oracle 是主 DCLDE 干净背景上的 `E(N_t)`，与图 3 中未标注 SanctSound WAV 消融的 oracle 不是同一实验。

针对 `E(N_t)` 丢失时间结构和相位的问题，另做**全波形 Oracle 抄答案校验**。固定已经训练好的真正 BSR，只拟合逐采样点的三系数头 `w_b·BSR(x)+w_x·x+w_c·c`。它从 `(1,0,0)` 开始，精确复现 BSR；用 8192 个训练混合的波形 MSE 最小二乘全局最优解分别拟合零波形、错时背景波形和当前真实 `N_t` 波形。测试仍是上表的 4096 个混合，逐例核对 ID、SNR、混合分数和 BSR 输出。

```powershell
python -m source.dclde_full_oracle_experiment --device cuda
```

真实 `N_t` 学得的系数为数值精度内的 `(0,1,−1)`，因此输出等于 `x−N_t=s_t`，最大绝对波形误差 `5.96e−8`。测试 SI-SDRi：原 BSR `9.631 dB`，零波形头 `9.231 dB`，错时背景头 `9.231 dB`，真实背景头和解析差分均为 `80.755 dB`（受 SI-SDR 数值底噪限制）；真实背景相对 BSR 的日期分块配对差为 `+71.124 dB`，95% 区间 `[69.945,72.081]`。逐例数据和完整系数见 [results/dclde_full_background_oracle_samples.csv](results/dclde_full_background_oracle_samples.csv) 与 [results/dclde_full_background_oracle.json](results/dclde_full_background_oracle.json)。这是有意非因果、不同接口的信息量校验，不能据此宣称 30 频带 GRU 预测能提供当前完整波形，也不能将其与六路匹配 FiLM 对照混为一谈。

### Oracle 信息阶梯

在相同冻结 BSR、三系数融合头、8192 个训练样本和匹配的 4096 个测试样本上，只压缩当前真实背景条件。512 点 STFT 提供 257 个频率 bin 和 64 ms 帧步长；对比等宽频率组、对数频带、逐帧至整段 5 秒的功率平均，以及真实背景相位和混合相位。每级分别在训练集求最小二乘最优系数，测试时严格逐例核对旧主实验的 ID、SNR、混合分数和 BSR 输出。

```powershell
python -m source.dclde_information_ladder --device cuda
python -m source.plot_oracle_ladder
```

| 当前背景条件 | 真实相位：相对 BSR Δ dB [日期区间] | 混合相位：相对 BSR Δ dB [日期区间] |
| --- | ---: | ---: |
| 完整 257 bin、逐帧 | +71.124 [69.945, 72.081] | +11.164 [10.889, 11.594] |
| 完整 257 bin、1 秒平均 | +1.948 [1.544, 2.667] | +0.585 [0.285, 1.132] |
| 完整 257 bin、5 秒平均 | +1.139 [0.799, 1.487] | +0.249 [−0.044, 0.495] |
| 等宽 128 组、逐帧 | +4.537 [3.118, 6.267] | +2.067 [0.969, 3.386] |
| 等宽 64 组、逐帧 | +0.471 [−0.494, 1.602] | −0.623 [−1.404, 0.259] |
| 对数 128 带、逐帧 | +0.261 [−0.599, 1.313] | −0.529 [−1.257, 0.319] |
| 对数 30 带、逐帧 | +0.117 [−0.671, 1.084] | −0.672 [−1.323, 0.086] |
| 对数 30 带、5 秒平均 | −0.025 [−0.401, 0.493] | −0.516 [−0.783, −0.150] |

**定位**：在这条固定的波形答案通路内，等宽组从 128 降到 64 时失去可确认的留出收益；对数分带在名义 128 带（其中只有 95 带含 STFT bin）已经失效。保持 257 bin 时，单独将功率汇总到 1 秒仍有可靠收益，因此频率平均及分带规则比时间平均更关键。最粗的“30 带/5 秒”与论文原始 `E(N_t)` 相似，但不是相同的数值定义。该阈值只针对这个线性融合接口，不能作为信息论极限，也不能代替 FiLM 重新训练。完整 29 级结果及逐例数据见 [results/dclde_information_ladder.json](results/dclde_information_ladder.json)、[results/dclde_information_ladder_samples.csv](results/dclde_information_ladder_samples.csv)；矢量图见 [paper/figures/oracle_information_ladder.pdf](paper/figures/oracle_information_ladder.pdf)。

### 128 组逐帧 Oracle 残差分离器

只使用当前真实干净背景作为 Oracle，不训练 GRU。对 `x_t` 与 `N_t` 做 512 点 STFT、128 点帧移，在每帧按 128 个等宽频率组计算平均功率；由 `log|B_t|` 和显式创新 `R_t=log|X_t|−log|B_t|` 组成 256 维逐帧图。该图插值到 BSR 帧网格，再通过零初始化线性投影加到混合声音得到的子带 token，送入轴向注意力。这是显式 `[X,B,R]` 的残差形式，**没有 FiLM 或历史预测**。训练前的波形输出与真正无条件 BSR 完全一致。

```powershell
python -m source.dclde_oracle_residual_experiment --device cuda
```

新模型与已有真正 BSR 共享训练/验证/测试混合、迁移骨干初值、种子、批次顺序、损失、优化器及余弦日程；两者均执行 6000 次更新，每 100 步验证并选择最高分 checkpoint。BSR 选中 5300 步，Oracle 模型选中 6000 步。**同一 4096 个留出样本**的目标 SI-SDRi 分别为 `9.631` 和 `16.767 dB`，逐例配对提升 `+7.136 dB`，按三个录音日分块的 95% 区间 `[6.970,7.282]`；三个日期各自的差值为 `+7.179/+7.282/+6.970 dB`。这满足事先提出的“至少稳定 +0.5 dB 且区间排除零”判据。

汇总见 [results/dclde_oracle_residual_128.json](results/dclde_oracle_residual_128.json)，逐例结果见 [results/dclde_oracle_residual_128_samples.csv](results/dclde_oracle_residual_128_samples.csv)，完整训练轨迹见 [results/dclde_oracle_residual_128_training.json](results/dclde_oracle_residual_128_training.json)，选中模型见 [models/oracle_residual_128.pt](models/oracle_residual_128.pt)。这是**当前背景真值**与显式创新接口的组合可行性结果；接口增加了 163,840 个可训练投影参数，因此这两路比较还不能单独量化背景信息与额外容量各自的作用，也不能代表历史预测条件的效果。

分离模型每个条件最多训练 10,000 次更新、至少 5,000 次，并依据固定验证集早停及选取 checkpoint；预测器最多 5,000 次更新、至少 2,500 次。正式比较使用相同留出日、相同 `(N_t,s_t)` 测试样本。其它 FiLM/拼接/交叉注意力、时间尺度、SanctSound TOL 和未标注 WAV 强度实验均为消融，不是主证据。

训练完成后生成论文图和强度结果曲线：

```powershell
python -m source.make_paper_figures
```

随后从仓库根目录运行本地实用前端（默认加载主模型）：

```powershell
python -m source.web
```

浏览器打开 `http://127.0.0.1:8000`。前端接受当前混合 WAV，以及此前 60 秒干净历史 WAV（自动提取特征）或至少 12 行事件掩蔽特征 CSV；模型文件为 `models/practical_film.pt`。历史 WAV 应由用户确保无事件或预先清理，不能把含鲸声的历史当作训练定义中的干净先验。

既有真实事件无参考评估包含 103 个拥有完整干净历史的测试鲸声。该历史实验中的旧“BS-RoFormer”检查点实际上是零条件 FiLM，对照须据此解读：零条件/GRU-Forecast 的背景事件概率为 `.786/.713`，事件声轨概率为 `.900/.902`，区间外相对误差为 `.230/.206`。按录音日分块，事件概率差的区间仍跨零，因此不能宣称真实事件语义分离已有确定优势。完整数据见 [results/dclde_real_event_results.json](results/dclde_real_event_results.json)；真正无条件 BSR 的全参考比较由本次六路 Oracle 实验提供。引用文献 PDF 与目录说明位于 [ref](ref)。

## 当前主实验：过去 60 秒的高分辨率因果残差

这一步把上面的 128 组逐帧 Oracle 真值图换成因果先验。过去 12 个 5 秒窗各形成 `79×128` 的对数幅度图；历史中的事件标注窗由同一过去 60 秒内最近的干净窗填补。预测器为小型残差 GRU，输出未来 5 秒完整 `79×128` 图。Persistence 直接沿用最后一个干净历史图。Zero 使用训练集固定逐频组均值图；Shuffled 从另一录音日抽取标注干净背景图。验证集只有一天，因此 Shuffled 的验证条件从训练日抽取。五路残差模型均按 `log|X|−B̂` 构造显式创新，使用完全相同的适配器；预测器先训练并冻结。4,096 个测试混合、BSR 骨干初值、合成目标、分离器 6,000 步预算与验证选点规则保持匹配。

```powershell
python -m source.dclde_causal_highres_experiment cache
python -m source.dclde_causal_highres_experiment predictor
python -m source.dclde_causal_highres_experiment forecasts
python -m source.dclde_causal_highres_experiment zero
python -m source.dclde_causal_highres_experiment shuffled
python -m source.dclde_causal_highres_experiment persistence
python -m source.dclde_causal_highres_experiment forecast
python -m source.dclde_causal_highres_experiment summary
```

缓存保存在被 Git 排除的 `data/dclde2013/`，两张逐帧图数组合计约 3.5 GB；重新构建需要本机 DCLDE WAV、干净背景清单、已有特征缓存及迁移骨干。结果和检查点如下。

| 条件 | 留出目标 SI-SDRi (dB) | 相对 BSR (dB) |
| --- | ---: | ---: |
| BSR | 9.631 | 0 |
| Zero-Residual | 12.156 | +2.525 |
| Shuffled-Residual | 11.944 | +2.313 |
| Persistence-Residual | 14.295 | +4.664 |
| GRU-Forecast-Residual | 14.351 | +4.720 |
| Oracle-Residual | 16.767 | +7.136 |

Zero 相对 BSR 提高 `+2.525 dB`，Shuffled 提高 `+2.313 dB`，说明不能把因果模型相对 BSR 的全部收益归因于历史信息。Zero 相对 BSR 的差异同时包含适配器参数和创新通路提供的当前混合对数谱，尚不能只量化“容量”作用。保持接口相同后，Persistence 相比 Zero 高 `+2.138 dB`，按三个录音日配对的 95% 区间为 `[1.775,2.377]`；相比错日 Shuffled 高 `+2.351 dB`，区间 `[1.571,2.661]`。三个留出日均同向，支持正确时刻的历史背景有额外价值。GRU 背景图的测试对数幅度 MAE 为 `0.5308`，Persistence 为 `0.5386`；Forecast 相对 Persistence 仅高 `+0.056 dB`，区间 `[0.036,0.074]`。因此 GRU 的增益真实而很小。下节独立报告真实事件上的迁移结果。

完整汇总见 [results/dclde_causal_highres_comparison.json](results/dclde_causal_highres_comparison.json)，逐例配对分数见 [results/dclde_causal_highres_test_samples.csv](results/dclde_causal_highres_test_samples.csv)；冻结预测器及五路残差检查点位于 `models/`，包括 `highres_gru_128.pt`、`zero_residual_128.pt`、`shuffled_residual_128.pt`、`persistence_residual_128.pt`、`forecast_residual_128.pt`、`oracle_residual_128.pt`。

### 128 频组残差模型的真实鲸声迁移

使用与前期 FiLM 评估完全相同的 103 个测试鲸声片段、冻结事件分类器及恢复干预，不用真实隔离声轨训练或选点。每段鲸声的 60 秒条件历史只取自同一 WAV 中严格早于当前的标注干净 5 秒窗；6 段历史各缺一窗，用这 60 秒内最近的干净窗填补。Shuffled 从另一测试录音日抽取标注干净背景；Forecast 现场运行冻结的 128 频组 GRU，条件不读取当前鲸声。

```powershell
python -m source.dclde_real_residual_eval --device cuda --batch-size 8
```

| 残差条件 | $p(x)$ | $p(\hat N)\downarrow$ | $p(\hat s)\uparrow$ | 恢复单调率$\uparrow$ | 区间外相对改动$\downarrow$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero | .909 | .891 | .802 | .990 | .062 |
| Shuffled | .909 | .781 | .910 | .989 | .078 |
| Persistence | .909 | .829 | .849 | .992 | .037 |
| GRU Forecast | .909 | .838 | .842 | .989 | .036 |

Persistence 相对 Zero 的背景事件概率下降 `.0624`（按三个录音日配对的 95% 区间 `[-.0682,-.0547]`），事件声轨概率上升 `.0469`（`[.0198,.1006]`），区间外改动下降 `.0255`（`[-.0386,-.0077]`）。但错日 Shuffled 在两项分类器事件概率上优于 Persistence/Forecast，同时区间外改动明显更大：Persistence 减 Shuffled 的区间外差为 `-.0408`（`[-.0519,-.0320]`）。单调率四路接近饱和。因此不能宣称真实目标上正确历史条件全面胜出；分类器可能奖励激进的事件提取，而因果条件在背景保留上更好。逐例数据和全部配对区间见 [results/dclde_real_residual_event_results.csv](results/dclde_real_residual_event_results.csv) 与 [results/dclde_real_residual_event_results.json](results/dclde_real_residual_event_results.json)。

### 冻结分类器底噪与合成鲸声覆盖诊断

此诊断保持上述 103 个真实测试事件和冻结分类器不变。每个真实事件取同一 WAV 中此前最近的标注干净 5 秒窗作为配对背景，并另从测试日按日期比例确定性抽取 4096 个标注干净窗。分类器只接收与分离器输出长度一致的前 9984 个采样点。以每个配对背景为底，按训练时的五种鲸声合成类型和 $[-18,12]$ dB 强度范围生成 32 个样本，共 3296 个。

```powershell
python -m source.diagnose_target_gap --device cuda --broad-clean-count 4096 --synths-per-event 32
```

| 冻结分类器输入 | 数量 | 平均事件概率 | 中位事件概率 |
| --- | ---: | ---: | ---: |
| 广泛抽取的标注干净背景 | 4096 | .153 | .093 |
| 与真实事件同 WAV 配对的标注干净背景 | 103 | .176 | .093 |
| 真实鲸声混合 | 103 | .909 | — |
| 干净背景 + 合成鲸声 | 3296 | .407 | .150 |

四路分离背景的平均事件概率 `.781–.891` 均明显高于同 WAV 干净背景；逐例相减的平均差为 Shuffled `.606`、Persistence `.653`、Forecast `.662`、Zero `.715`。因此分类器自身的非零底噪解释不了高响应，但没有真实隔离声轨，不能把这个概率差直接换算成鲸声残留能量；分离伪影和漏标也可能触发分类器。

覆盖检查使用该分类器冻结的 64 维倒数第二层激活：每个有事件片段的激活减去同 WAV 配对干净片段的激活，按合成样本逐维标准化，再计算欧氏最近邻距离。真实事件彼此的留一最近邻中位距离为 `.233`。若直接用全部 3296 个合成样本，真实到合成的中位距离为 `.262`，只有 `3.9%` 超过真实最近邻距离的第 95 百分位；但候选数不匹配。每次只取 103 个合成样本、每个真实事件对应一个，并重复 32 次后，真实到合成的中位最近邻距离为 `.483`，是 `.233` 的 `2.08` 倍；超过真实第 95 百分位阈值的比例中位数为 `51.5%`。五类合成鲸声的平均分类器概率也相差很大：上扫 `.700`、下扫 `.134`、拱形 `.338`、S 形 `.699`、脉冲包 `.161`。这些结果表明当前合成分布在**等量、分类器表征**下较稀疏，支持 synthetic-to-real 覆盖不足的假设；但真实事件相对前一干净窗的背景变化也进入了嵌入差，且分类器表征并非孤立鲸声分布距离，因此不能证明覆盖缺口是分离失败的唯一原因。逐例数据见 [results/dclde_target_gap_diagnostic.csv](results/dclde_target_gap_diagnostic.csv)，全部诊断与抽样设置见 [results/dclde_target_gap_diagnostic.json](results/dclde_target_gap_diagnostic.json)，合成抽样明细见 [results/dclde_target_gap_synthetic_samples.csv](results/dclde_target_gap_synthetic_samples.csv)。

### 训练日经验目标池与覆盖验收

目标池只读取训练日 `2009-03-28` 至 `2009-03-30` 的真实标注鲸声和同一 WAV 内事件前后的标注干净背景。对当前事件的 STFT 幅度，估计前后干净窗的局部背景对数幅度插值，取事件标注内、40–950 Hz 的正超额能量，平滑生成 soft mask，并用当前混合相位逆变换得到伪鲸声。该声轨含参考背景估计误差，明确不是干净 GT。2798 个候选事件中有 2616 个通过前后干净窗与声学对比度检查；筛选完全不调用最终冻结分类器。

同一批训练事件还提供时长、频率上下界、频移与斜率、曲率、带宽、谐波峰数与衰减、包络峰值位置、频率抖动、脉冲间隔和标注事件数的经验联合样本。程序合成路线抽取一个真实鲸声轮廓并轻微随机化音高、时长、抖动、相位、谐波幅度、间隔与起点，同时保留原有的船舶音调与空化噪声；它在目标池中占 50%，其中船噪占该路线的 $4/7$，使最终鲸声/船噪总体比例与旧七类训练池一致。另一半直接复用软掩码伪鲸声并随机化相位与时刻。真实测试日绝不参与参数拟合或伪声轨提取。

```powershell
python -m source.empirical_target_pool
python -m source.diagnose_target_gap --device cuda --target-pool empirical_50_50
python -m source.plot_empirical_target_acceptance
```

验收时冻结分类器仅用于**诊断**，没有参与合成器拟合或训练样本筛选。沿用同一 103 个真实事件、32 次每轮 103 个合成候选的协议；新池每轮也保持两路约 50/50。等量最近邻距离从旧目标池 `.483` 降至新目标池 `.322`；超出真实近邻第 95 百分位阈值的比例中位数从 `51.5%` 降至 `3.4%`，32 轮均改善。新池合成混合的平均事件概率为 `.647`，其中包含船噪的程序合成路线为 `.352`、伪真实为 `.942`；这只是冻结分类器的外部诊断，不能充当波形真值。逐事件形态参数见 [results/dclde_train_whale_morphology.csv](results/dclde_train_whale_morphology.csv)；完整分布与抽样明细见 [results/dclde_train_target_pool.json](results/dclde_train_target_pool.json)、[results/dclde_empirical_target_gap_diagnostic.json](results/dclde_empirical_target_gap_diagnostic.json) 和 [results/dclde_empirical_target_acceptance.json](results/dclde_empirical_target_acceptance.json)。另提供可放大的[覆盖验收矢量图](paper/figures/dclde_empirical_target_acceptance.pdf)。

覆盖验收通过后才重训两种因果 128 频组残差分离器；历史 GRU 和冻结事件分类器均不重训。两路使用相同初始骨干、6000 更新、每 500 更新在相同 4096 个验证背景上选最佳权重。随后在原有、完全不变的 4096 个合成测试混合上审计旧目标分布的退化，再对相同 103 个真实鲸声做冻结分类器与区间外保留评估：

```powershell
python -m source.dclde_causal_highres_experiment persistence --device cuda --target-pool empirical_50_50
python -m source.dclde_causal_highres_experiment forecast --device cuda --target-pool empirical_50_50
python -m source.audit_empirical_separator
python -m source.evaluate_empirical_residual --device cuda
```

| 条件与目标池 | 不变旧合成测试 SI-SDRi (dB) | 真实鲸声 $p(\hat N)\downarrow$ | $p(\hat s)\uparrow$ | 标注区间外相对改动$\downarrow$ |
| --- | ---: | ---: | ---: | ---: |
| Persistence，旧程序目标 | 14.295 | .829 | .849 | .037 |
| Persistence，新 50/50 目标 | 12.195 | .652 | .916 | .058 |
| Forecast，旧程序目标 | 14.351 | .838 | .842 | .036 |
| Forecast，新 50/50 目标 | 11.731 | .725 | .895 | .043 |

在同一 103 个真实鲸声上，新 Persistence 相对旧版的 $p(\hat N)$ 差为 `-.1766`（按三个录音日配对的 95% 区间 `[-.1920,-.1479]`），$p(\hat s)$ 差为 `+.0668`（`[.0414,.1166]`）；但区间外改动也增加 `+.0215`（`[.0151,.0324]`）。新 Forecast 对旧版相应为 `-.1129`、`+.0527` 和 `+.0068`。新 Persistence 的事件响应更强，新 Forecast 的区间外保留更好；分类器概率不等于真实隔离声轨质量。两种新模型在**旧合成分布**上的 SI-SDRi 分别下降 2.100 和 2.620 dB，说明目标域适配存在明显旧域退化。中间试过只用鲸声的经验池：Persistence 的真实事件概率同样改善，但旧船舶音调 SI-SDRi 降到 `-1.925 dB`；最终版本保留船噪后恢复为 `5.350 dB`。完整逐类受控审计见 [results/dclde_empirical_residual_legacy_test_audit.json](results/dclde_empirical_residual_legacy_test_audit.json)，真实逐例配对结果见 [results/dclde_empirical_residual_real_event_results.csv](results/dclde_empirical_residual_real_event_results.csv) 与 [results/dclde_empirical_residual_real_event_results.json](results/dclde_empirical_residual_real_event_results.json)。

最终分离器检查点随仓库提供：[Persistence](models/empirical_persistence_residual_128.pt)、[Forecast](models/empirical_forecast_residual_128.pt)；冻结的 [128 频组 GRU](models/highres_gru_128.pt) 保持与旧实验同一文件。中间鲸声专用 [Persistence 检查点](models/empirical_whale_only_persistence_residual_128.pt)仅用于解释船噪退化，不作为通用版本。约 52 MB 的训练日伪声轨缓存位于忽略目录，可用前述 `source.empirical_target_pool` 命令重建；没有引入测试日真实事件到目标池。

### 当前残差模型的分离图与训练充分性检查

论文 Fig. 5 已改成**当前 128 频组残差模型**，同时显示无条件 BSR、经验目标池训练的 Persistence/Forecast 与同一个合成目标真值。图从固定的 96 个 $-12$ dB 留出测试样本中按三模型 SI-SDRi 的联合中位数选例，不选最好看的样本。真值目标低于海洋背景 20 dB 的时频格点里，96 例的事件声轨/海洋背景对比度中位数依次为 BSR `-1.37`、Persistence `-31.44`、Forecast `-36.46` dB；所选样本中，仍有局部竖向伪影。见[可放大 PDF](paper/figures/real_audio_visualization.pdf)、[PNG 预览](paper/figures/real_audio_visualization_preview.png)和[数值及选样规则](paper/figures/real_audio_visualization.json)。这个受控合成图不能证明真实鲸声的隔离质量。

小集过拟合对照保持样本和混合确定：128 个旧程序目标训练 3000 步，训练 SI-SDRi 从 `13.85` 升至 `17.20` dB；128 个经验目标训练 20000 步，从 `10.57` 升至 `15.56` dB，最后几千步增幅已明显放缓，而非目标区对比度中位数从 `-32.85` 降至 `-60.68` dB。经验目标中的校准程序路线从 `8.20` 升至 `13.56` dB，伪真实路线从 `12.95` 升至 `17.56` dB。模型确实继续学习复杂目标，但剩余训练误差已不能简单归因于背景泄漏；波形损失与 SI-SDRi 也不完全同步。详见[小集审计](results/dclde_residual_overfit_empirical_audit.json)。

从旧程序 Persistence 强权重做 `0% → 25% → 50%` 伪真实课程微调，学习率由 `5e-5` 降至 `5e-6`。训练日的伪鲸声分成互不重叠的 80% 拟合集与 20% 验证集；另保留固定旧程序验证集，各 512 个样本。联合 SI-SDRi 选中第 6500 步（25% 伪真实）：两套验证集从 `16.26/11.12` 升至 `16.44/11.84` dB。独立追加的 50/50 阶段跑满早停前的 4000 步，虽然更压低非目标区能量，联合 SI-SDRi 未继续提高，因此未用真实测试鲸声选点。选中模型在不变旧合成测试上得到 `14.421` dB，高于原 Persistence 的 `14.295` dB；103 个真实事件的 $p(\hat N)/p(\hat s)$ 为 `.819/.823`，仍明显不及直接 50/50 训练的 `.652/.916`，所以课程微调没有同时解决旧域与真实语义指标。相同小集上增加少量 SI-SDR 损失可多得 `0.40 dB`，但对应的完整数据 50/50 阶段再训 4000 步并未超过同一验证起点，故未选用。旧 Fig. 5 的大片 0 dB 泄漏在新模型中已消失，且更强的背景抑制伴随 SI-SDRi 下降；本轮暂不增加额外泄漏损失，以免把这一已观察到的取舍推得更远。完整结果见[课程轨迹](results/dclde_residual_curriculum.json)、[50/50 追加轨迹](results/dclde_residual_curriculum_resume.json)、[损失对照](results/dclde_residual_objective_probe.json)、[完整数据损失对照](results/dclde_residual_curriculum_resume_sisdr.json)、[不变测试](results/dclde_refined_persistence_legacy_test.json)、[真实事件评估](results/dclde_refined_persistence_real_event_results.json)、[选中模型](models/refined_persistence_residual_128.pt)及[收敛矢量图](paper/figures/residual_training_diagnostic.pdf)。诊断没有重训 GRU 或分类器，也没有新增数据下载。

```powershell
python -m source.plot_residual_qualitative
python -m source.refine_residual_separator overfit --pool legacy --examples 128 --steps 3000
python -m source.refine_residual_separator overfit --pool empirical --examples 128 --steps 10000
python -m source.refine_residual_separator continue-overfit --pool empirical --examples 128 --steps 10000
python -m source.refine_residual_separator audit-overfit --pool empirical --examples 128
python -m source.refine_residual_separator curriculum --steps 20000
python -m source.refine_residual_separator resume --steps 12000
python -m source.refine_residual_separator objective-probe --pool empirical --examples 128 --steps 2000
python -m source.refine_residual_separator resume-sisdr --steps 4000
python -m source.refine_residual_separator test
python -m source.evaluate_empirical_residual --only persistence --checkpoint models/refined_persistence_residual_128.pt --output results/dclde_refined_persistence_real_event_results.json
python -m source.plot_refinement_learning
```

### 真实标注事件弱监督微调（独立检测器）

新增的逐帧训练检测器只用 DCLDE 训练日事件时间标注拟合，验证日逐帧 AP 为 `0.901`。它是扩张时序卷积网络，与原有冻结的整段**最终评估**分类器不同。弱监督分离器每步混合 2 个有真值的训练日 synthetic/pseudo-real 样本与 2 个真实标注事件；真实事件背景声轨由 `mixture − event stem` 定义，标注区间外另加保留约束。合格的真实事件为训练日 180 段、验证日 47 段；103 段测试日事件和最终评估分类器均不参与训练或选点。

```powershell
python -m source.weak_event_detector --epochs 30
python -m source.weak_real_finetune --steps 3000 --validate-every 250
python -m source.evaluate_weak_real
python -m unittest discover -s tests -q
```

验证日按预先设定的事件响应、区间外改动和旧合成 SI-SDRi 约束选中第 250 步。**选定后只评估一次**的同批 103 段真实测试鲸声：独立分类器给出 `p(N̂) 0.652 → 0.408`、`p(ŝ) 0.916 → 0.978`、区间外相对改动 `0.058 → 0.061`。配对同 WAV 事件前干净背景的平均概率为 `0.176`；逐例干净归一化 logit 残留量从 `3.606` 降到 `1.389`。但是，不变的 4096 个旧合成测试混合 SI-SDRi 从 `12.195` 降到 `10.323 dB`。这说明真实事件弱监督改善了独立语义代理指标，尚未达到干净背景水平，也有明确的旧目标域损失；真实鲸声隔离声轨仍没有真值。

复现文件：[训练检测器](models/weak_train_frame_detector.pt)、[选中的分离器](models/weak_real_persistence_residual_128.pt)、[训练检测器验证结果](results/dclde_weak_train_detector.json)、[验证日选点](results/dclde_weak_real_finetune_selection.json)、[全部验证轨迹](results/dclde_weak_real_finetune_training.json)、[封存测试审计](results/dclde_weak_real_persistence_audit.json)及[103 段逐例结果](results/dclde_weak_real_persistence_event_results.csv)。
