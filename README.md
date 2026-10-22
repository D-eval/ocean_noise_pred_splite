# Forecast-conditioned ocean source separation

本仓库把两个原型连接成一个可训练系统：用严格因果的历史海洋声景预测当前噪声，随后把预测频谱作为 BS-RoFormer 的条件，分离叠加在当前背景上的合成声源。代码同时保留表示消融，并完成三种条件注入消融：

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

打开 `http://127.0.0.1:8000`，上传至少 24 行历史 TOL CSV 和当前 WAV，即可试听、下载目标信号与海洋背景。浏览器数据只发送到本机后端。10 秒音频在 RTX 3060 上的端到端 RTF 为 0.043。

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

## 真实标注事件与无参考评估

DCLDE 2013 连续录音保存在 `data/dclde2013/`（约 2.3 GiB，已由 `.gitignore` 排除）。实验按录音日划分数据，用确定的露脊鲸上扫声作正样本；“干净背景”会排除确定及不确定事件前后 2 秒。训练阶段只把合成上扫声加入干净真实海洋片段，真实事件只用于最终无参考评估。

```powershell
python -m source.event_classifier --prepare-only
python -m source.event_classifier --epochs 30 --batch-size 64 --device cuda
python -m source.dclde_separation_experiment --steps 2500 `
  --minimum-steps 1500 --batch-size 4 --device cuda
```

冻结分类器在按日留出的测试集上达到 AUROC `0.9704`、AP `0.9738`、F1 `0.9196`。严格因果条件由当前片段之前的八个 8 秒窗构成；预测器的验证 MAE 为 `1.921 dB`，优于一步持续性基线的 `2.081 dB`（改善 `7.67%`）。无条件 BS-RoFormer 与 FC-BSR 使用相同真实干净背景、合成上扫声、随机种子、分离器初值、训练步数和验证选模规则，验证 SI-SDRi 分别为 `13.871` 和 `13.925 dB`。

在两种模型共同可用的 1225 个按日留出的真实事件片段上，输入事件概率均为 `0.855`。BS-RoFormer/FC-BSR 的背景声轨事件概率为 `0.523/0.507`，事件声轨概率为 `0.901/0.901`，恢复单调率为 `98.61%/98.50%`，区间外相对改动为 `9.56%/10.22%`。按三个独立录音日做配对 bootstrap 后，FC-BSR 相对基线的背景事件概率差为 `−0.0160`（95% CI `−0.0210` 至 `−0.0132`），而区间外改动增加 `0.0066`（`0.0033` 至 `0.0078`）。因此预测条件确实加强了真实鲸声的语义移除、没有削弱事件声轨证据，但伴随小而一致的背景保真代价，不能表述为无条件胜出。恒等输出和“全部归入事件”两个退化对照也同时报告。训练摘要见 [results/dclde_training_results.json](results/dclde_training_results.json)，完整评估见 [results/dclde_real_event_results.json](results/dclde_real_event_results.json)，逐样本数据见 [results/dclde_real_event_results.csv](results/dclde_real_event_results.csv)，矢量图见 [paper/figures/dclde_reference_free.pdf](paper/figures/dclde_reference_free.pdf)。引用文献 PDF 与目录说明位于 [ref](ref)。
