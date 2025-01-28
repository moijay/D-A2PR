# A2PR项目修改日志

## 修改概述
本次修改的主要目标是将原始A2PR算法中的VAE（变分自编码器）替换为Diffusion Model（扩散模型）用于动作生成。修改涉及三个主要文件：main.py、A2PR.py和config.py。

## 具体修改内容

### 1. A2PR.py 的修改

#### A. 类的替换
1. 删除原有的 `VAE` 类
2. 新增 `DiffusionModel` 类，包含以下方法：
   - `__init__`: 初始化扩散模型参数
   - `forward`: 实现前向扩散过程
   - `sample`: 实现反向扩散过程生成动作

#### B. A2PR类的修改
1. 参数修改：
   - 将 `vae_weight` 参数改为 `diffusion_weight`
   - 移除 VAE 相关的初始化参数

2. 初始化部分修改：
   - 将 `self.vae` 替换为 `self.diffusion`
   - 将 `self.vae_optimizer` 替换为 `self.diffusion_optimizer`
   - 更新 `self.models` 字典中的相关项

3. train方法修改：
   - 移除 VAE 训练相关代码
   - 新增扩散模型训练代码
   - 更新损失函数计算方式
   - 修改 tb_statics 中的统计信息

### 2. main.py 的修改

1. 信息标识修改：
   - 将 `info = 'A2PR'` 改为 `info = 'A2PR_Diffusion'`

2. 结果目录路径修改：
   - 将 `-vae_weight` 改为 `-diffusion_weight`
   - 使用 `args.diffusion_weight` 替代 `args.vae_weight`

### 3. config.py 的修改

1. 默认算法名修改：
   - 将 `algorithm="A2PR"` 改为 `algorithm="A2PR_Diffusion"`

2. 参数修改：
   - 移除 `--vae_weight` 参数
   - 新增 `--diffusion_weight` 参数
   - 新增扩散模型特定参数：
     ```python
     parser.add_argument("--num_timesteps", default=100, type=int)
     parser.add_argument("--beta_start", default=1e-4, type=float)
     parser.add_argument("--beta_end", default=0.02, type=float)
     ```

3. kwargs 字典修改：
   - 将 `"vae_weight": args.vae_weight` 改为 `"diffusion_weight": args.diffusion_weight`
   - 移除不必要的扩散模型特定参数（作为 DiffusionModel 的默认参数）

## 修改中的问题修复

### 1. 参数错误修复
- 修复了 `main.py` 中使用 `--vae_weight` 导致的未识别参数错误
- 移除了 `kwargs` 中导致 `A2PR` 初始化错误的多余参数

### 2. 代码优化
- 保持了 `DiffusionModel` 参数的默认值
- 简化了参数传递流程
- 确保了与原始代码框架的兼容性


