# Forecast-conditioned ocean source separation

本仓库把两个原型连接成一个可训练系统：用严格因果的历史海洋声景预测当前噪声，随后把预测表示作为 BS-RoFormer 的条件，分离叠加在当前背景上的合成声源。代码包含任务要求的三组消融：

1. `none`：不做噪声预测，直接训练条件置零的 BS-RoFormer；
2. `latent`：将 GRU 历史编码器的预测潜表示注入 band-split token；
3. `spectrum`：将预测的当前 TOL 频谱注入同一位置。

## 数据口径

默认使用 `third_party/soundscape/datasets` 中 SanctSound MB03 的原始 TOL CSV：MB03_01/02 训练、MB03_03 验证、MB03_04 测试。条件历史截止到 `t-1`，`t` 时刻频谱只作为预测监督和当前背景的构造依据，因此预测器看不到未来。

公开的 TOL 产品没有原始波形相位。训练中的“当前海洋噪声”是按当小时实测 TOL 包络生成的确定性随机相位代理波形，不是恢复出来的现场录音。它适合验证联合训练和消融逻辑；论文级现场可用性结论还必须在时间对齐的原始水听器波形上复验。

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

已训练的实用配置使用 23.5 万参数、1500 次更新和 −12 至 +12 dB 的训练范围。仓库包含约 1 MB 的检查点：

```powershell
python -m source.web
```

打开 `http://127.0.0.1:8000`，上传至少 24 行历史 TOL CSV 和当前 WAV，即可试听、下载目标信号与海洋背景。浏览器数据只发送到本机后端。10 秒音频在 RTX 3060 上的端到端 RTF 为 0.043。

输出保存在 `experiments/ablation/<variant>/`。模型权重被 `.gitignore` 排除；小型 JSON/CSV 指标可选择性纳入版本控制。总表 `ablation_summary.json` 给出两种条件相对无条件基线的配对 SI-SDRi 增益、bootstrap 95% 区间和逐样本胜率。

## 方法摘要

历史频谱经 GRU 编码为潜变量并解码为下一小时频谱。联合目标为

`L = L_wave+MRSTFT + 0.2 * L_forecast`。

条件向量经 LayerNorm 和线性映射变为每个 BS-RoFormer 子带的偏置，在 band split 后、轴向 Transformer 前相加；因此分离损失可以反向更新预测器。合成器实现谐波、带限噪声、包络和 LFO，并记录每个样本的参数。详细定义、限制和正式实验协议见 [paper/main.tex](paper/main.tex)。

## 已完成的 smoke 结果

300 步小模型实验中，无条件基线、潜表示条件、频谱条件的测试 SI-SDRi 分别为 3.75、3.44、4.15 dB。潜表示相对基线下降 0.31 dB（配对 95% CI：−0.43 至 −0.20），预测频谱提高 0.39 dB（0.17 至 0.61）；三种方案均能快于实时运行。这个结果只支持 TOL 代理波形上的“显式预测频谱”接口，不能外推为现场原始音频结论。完整未四舍五入结果见 [results/smoke_ablation.json](results/smoke_ablation.json)。

## 实用训练与强度测试

1500 步实用模型总体 SI-SDRi 为 5.74 dB（95% CI：5.04–6.47），通过预注册工程门槛。−12/−6/0/+6/+12 dB 五档输入的 SI-SDRi 为 6.55/6.32/5.73/3.74/−0.14 dB；+12 dB 档的绝对目标 SI-SDR 为 11.89 dB，说明高强度输入已接近“无需处理”的区域。

同规模、同训练量的无条件 BS-RoFormer 基线为 6.02 dB；频谱条件相对基线的配对差为 −0.28 dB（95% CI：−0.69 至 0.14），没有显著提升。因此当前结论是“系统达到代理数据上的工程可用性，但长期训练后预测条件未超过强基线”。详细结果见 [results/practical_results.json](results/practical_results.json)。

论文提供英文 [main.tex](paper/main.tex) 与中文 [main_zh.tex](paper/main_zh.tex) 两份源文件，以及已渲染的[英文 PDF](output/pdf/forecast_conditioned_ocean_separation_en.pdf)和[中文 PDF](output/pdf/forecast_conditioned_ocean_separation_zh.pdf)。
