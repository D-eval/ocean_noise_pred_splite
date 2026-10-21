# 海洋背景噪声矢量时频分析器

在仓库根目录双击 `open_ocean_noise_viewer.cmd`。脚本会启动本地计算服务并自动打开浏览器；黑色命令窗口需要保持开启，关闭窗口即停止服务。

在时频图中单击任意位置，会把该位置的横坐标作为试听起点，在图上显示橙色标记并播放对应的真实海洋录音。播放器可以选择 10、30、60 或 120 秒。

按住鼠标拖出矩形区域则用于放大。松开后，后端会读取该时间段对应的原始 8 kHz FLAC，按选中频段自适应选择 STFT 窗长，并以更细的时间步长和频率分辨率重新计算。图像由 SVG 矢量路径绘制，不使用 canvas 位图。

也可以在 PowerShell 中启动：

```powershell
F:\Applications\miniconda\python.exe -m source.day_noise_viewer --open
```

默认地址为 `http://127.0.0.1:8767/`。
