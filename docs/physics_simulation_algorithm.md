# 物理CT模拟算法详解

> **CT Simulation — Physics-Based Algorithm Documentation**
>
> 本文档详尽介绍本项目中 *真实物理CT模拟* 各模块的数学原理、算法流程和实现细节。

---

## 目录

1. [总体架构](#1-总体架构)
2. [X射线能谱生成](#2-x射线能谱生成)
3. [材料衰减数据库](#3-材料衰减数据库)
4. [多色X射线投影 (Beer-Lambert 定律)](#4-多色x射线投影-beer-lambert-定律)
5. [Radon 变换 (正投影)](#5-radon-变换-正投影)
6. [滤波反投影重建 (FBP / iRadon)](#6-滤波反投影重建-fbp--iradon)
7. [噪声模型 (Poisson 统计)](#7-噪声模型-poisson-统计)
8. [散射模拟](#8-散射模拟)
9. [运动模糊模拟](#9-运动模糊模拟)
10. [相对衰减值归一化](#10-相对衰减值归一化)
11. [压缩力学仿真](#11-压缩力学仿真)
12. [GPU 加速策略](#12-gpu-加速策略)
13. [简化模式 vs 物理模式对比](#13-简化模式-vs-物理模式对比)
14. [Future Works：提升物理模拟准度](#14-future-works提升物理模拟准度)
15. [参考文献](#15-参考文献)

---

## 1. 总体架构

物理模式 CT 模拟的完整数据流如下：

```
STL网格
  │
  ▼
┌──────────────┐
│  体素化       │  Voxelizer: 三角面片→二值体积
│  (Voxelizer)  │
└──────┬───────┘
       │  VoxelGrid (binary 0/1)
       ▼
┌──────────────────┐
│ 结构生成          │  晶格 / 随机缺陷 → 修改 VoxelGrid
│ (StructureModifier)│
└──────┬───────────┘
       │  VoxelGrid (含缺陷)
       ▼
┌──────────────────────────────────────────────┐
│          PhysicalCTSimulator                  │
│                                              │
│  1. SpectrumGenerator → XRaySpectrum         │
│  2. AttenuationDatabase → μ(E) 查表          │
│  3. 能量分箱 (Energy Binning)                │
│  4. 逐层模拟:                                │
│     ├─ Radon 变换 → 路径长度                  │
│     ├─ 多色衰减 (Beer-Lambert) → I_primary   │
│     ├─ 运动模糊 (可选)                        │
│     ├─ 散射叠加 (可选) → I_total             │
│     ├─ Poisson 噪声 → I_noisy                │
│     ├─ −log 变换 → Sinogram                  │
│     └─ iRadon (FBP) → 重建切片               │
│  5. 相对衰减值归一化                          │
└──────┬───────────────────────────────────────┘
    │  CTVolume (相对衰减值)
       ▼
┌──────────────────┐
│ 压缩仿真          │  弹性力学 → 位移场 → 变形体积 → 再次CT模拟
│ (CompressionManager)│
└──────┬───────────┘
       │  List[CompressionResult]
       ▼
   DICOM 导出 + 注释导出
```

**关键源文件映射：**

| 模块 | 文件 | 功能 |
|------|------|------|
| 物理模拟器 | `simulation/physics/physical_simulator.py` | 主协调器，CPU路径 |
| GPU模拟器 | `simulation/physics/gpu_simulator.py` | GPU批量处理 |
| 能谱生成 | `simulation/physics/spectrum.py` | Kramers定律 + 特征辐射 |
| 衰减数据库 | `simulation/physics/attenuation.py` | NIST XCOM μ/ρ 数据 |
| 散射模型 | `simulation/physics/scatter.py` | 卷积散射 + 运动模糊 |
| GPU Radon | `simulation/backends/radon_kernels.py` | GPU Radon/iRadon |
| CPU Radon | `simulation/backends/cpu_backend.py` | scikit-image 后端 |
| 弹性求解 | `simulation/mechanics/elasticity.py` | FDM Jacobi 求解器 |
| 压缩管理 | `simulation/mechanics/manager.py` | 多步压缩编排 |
| 材料库 | `simulation/materials.py` | 统一材料属性 |

---

## 2. X射线能谱生成

> **源文件**: `simulation/physics/spectrum.py`

### 2.1 物理背景

医学/工业CT使用的X射线管产生 **多色 (polychromatic)** X射线束，主要由两部分组成：

1. **轫致辐射 (Bremsstrahlung)**：电子在钨靶上减速产生的连续能谱
2. **特征辐射 (Characteristic radiation)**：钨原子内层电子跃迁产生的离散能量线

### 2.2 Kramers 定律

轫致辐射的连续能谱由 Kramers 定律近似描述：

$$
I(E) \propto Z \cdot \frac{E_{\max} - E}{E}
$$

其中：

- $E_{\max}$ = 管电压 (kVp)，即光子的最大能量
- $E$ = 光子能量 (keV)
- $Z$ = 靶材原子序数 (钨: $Z=74$)

代码实现（简化为Z=const,因为靶材不变）：

```python
# energies: [1, 2, ..., kVp] keV
bremsstrahlung[E] = (kVp - E) / E    # E < kVp
```

### 2.3 特征辐射峰

当管电压超过钨的K壳层结合能 (~69.5 keV) 时，在以下能量处叠加特征峰：

| 谱线 | 能量 (keV) | 相对强度 |
|------|-----------|---------|
| Kα₁ | 59.32 | ~50% of bremsstrahlung |
| Kα₂ | 57.98 | ~30% |
| Kβ  | 67.24 | ~15% |

```python
if kVp > 70:
    bremsstrahlung[59] += bremsstrahlung[59] * 0.5   # Kα₁
    bremsstrahlung[58] += bremsstrahlung[58] * 0.3   # Kα₂
    bremsstrahlung[67] += bremsstrahlung[67] * 0.15  # Kβ
```

### 2.4 铝滤过 (Filtration)

CT系统使用铝滤波器硬化射束（去除低能光子）。应用 Beer-Lambert 定律：

$$
I(E) = I_0(E) \cdot \exp\left(-\mu_{\text{Al}}(E) \cdot t\right)
$$

其中：

- $\mu_{\text{Al}}(E)$ = 铝的线性衰减系数 $= (\mu/\rho)_{\text{Al}}(E) \times \rho_{\text{Al}}$
- $\rho_{\text{Al}} = 2.7 \text{ g/cm}^3$
- $t$ = 滤过厚度 (mm)

铝的质量衰减系数通过对标准NIST数据的插值获得：

| E (keV) | $\mu/\rho$ (cm²/g) |
|---------|-------------------|
| 20 | 3.44 |
| 40 | 0.52 |
| 60 | 0.21 |
| 80 | 0.14 |
| 100 | 0.11 |
| 120 | 0.09 |
| 140 | 0.08 |

### 2.5 典型临床参数

| kVp | 滤过 (mm Al) | 平均能量 (keV) |
|-----|-------------|---------------|
| 80  | 2.0 | ~45 |
| 100 | 2.5 | ~55 |
| 120 | 2.5 | ~65 |
| 140 | 3.0 | ~70 |

最终输出为归一化的 `XRaySpectrum` 对象（$\sum \phi(E) = 1$），后续作为加权系数使用。

---

## 3. 材料衰减数据库

> **源文件**: `simulation/physics/attenuation.py`, `simulation/materials.py`

### 3.1 NIST XCOM 数据

衰减数据库存储20–150 keV范围内14个标准能量点的 **质量衰减系数** $\mu/\rho$ (cm²/g)，源自NIST XCOM数据库。

支持的材料包括：

| 材料 | 密度 (g/cm³) | μ/ρ @ 60 keV |
|------|-------------|-------------|
| Air | 0.001205 | 0.145 |
| Water | 1.00 | 0.206 |
| Soft tissue | 1.06 | 0.205 |
| Bone (cortical) | 1.85 | 0.248 |
| Aluminum | 2.70 | 0.209 |
| Titanium | 4.50 | 0.416 |
| Iron (Steel) | 7.87 | 0.598 |

### 3.2 线性衰减系数计算

对于任意能量 $E$，线性衰减系数通过插值和密度缩放获得：

$$
\mu(E) = \left(\frac{\mu}{\rho}\right)(E) \times \rho \quad [\text{cm}^{-1}]
$$

```python
class AttenuationTable:
    def get_mu(self, energy: float) -> float:
        mu_rho = np.interp(energy, self.energies, self.mu_rho)
        return mu_rho * self.density
```

### 3.3 频谱加权有效衰减

物理模拟器在初始化时计算水的 **有效衰减系数**（用于相对衰减归一化）：

$$
\mu_{\text{water}}^{\text{eff}} = \frac{\sum_E \phi(E) \cdot \mu_{\text{water}}(E)}{\sum_E \phi(E)}
$$

这是频谱加权的均值，代表等效单能束的衰减。

---

## 4. 多色X射线投影 (Beer-Lambert 定律)

> **核心物理**: `physical_simulator.py::_simulate_slice()` 和 `gpu_simulator.py::_simulate_slice_kernel()`

### 4.1 能量分箱 (Energy Binning)

为平衡精度和计算效率，将完整的1-kVp能谱离散化为 $N_{\text{bins}}$ 个能量箱（默认10箱）：

```python
energy_edges = np.linspace(E_min, E_max, N_bins + 1)
bin_centers = (energy_edges[:-1] + energy_edges[1:]) / 2
```

每个bin的权重为该范围内归一化光通量之和：

$$
w_i = \sum_{E \in \text{bin}_i} \phi(E), \quad \sum_i w_i = 1
$$

### 4.2 多色Beer-Lambert投影

对每个能量箱独立应用 Beer-Lambert 衰减定律，然后加权求和得到总透射强度：

$$
I_{\text{primary}} = \sum_{i=1}^{N_{\text{bins}}} w_i \cdot \exp\left(-\mu_{\text{obj}}^{(i)} \cdot L_{\text{obj}} - \mu_{\text{bg}}^{(i)} \cdot L_{\text{bg}}\right)
$$

其中：

- $L_{\text{obj}}$：X射线穿过物体区域的路径长度（由Radon变换给出）
- $L_{\text{bg}}$：穿过背景（空气）的路径长度
- $L_{\text{bg}} = L_{\text{total}} - L_{\text{obj}}$
- $L_{\text{total}} = 1.1 \times \max(L_{\text{obj}})$（射线束总路径）
- $\mu_{\text{obj}}^{(i)}, \mu_{\text{bg}}^{(i)}$：第 $i$ 个bin的物体/背景材料线性衰减系数

```python
for i in range(n_bins):
    L_obj = path_lengths                    # 从 Radon 变换获得
    L_bg  = max(max_path - L_obj, 0)
    attenuation = exp(-mu_obj[i] * L_obj - mu_bg[i] * L_bg)
    I_primary += w[i] * attenuation
```

> **注意**：这是多色投影的核心——它自然产生 **射束硬化 (beam hardening)** 伪影，因为低能光子被优先吸收，导致有效能量随穿透深度增加而升高。

---

## 5. Radon 变换 (正投影)

> **源文件**: `simulation/backends/radon_kernels.py`, `simulation/backends/cpu_backend.py`

### 5.1 数学定义

Radon变换将二维密度图 $f(x, y)$ 转换为沿不同角度的线积分（即投影数据/正弦图）：

$$
\mathcal{R}[f](\theta, t) = \int_{-\infty}^{\infty} f(t\cos\theta - s\sin\theta,\; t\sin\theta + s\cos\theta) \, ds
$$

其中：

- $\theta$：投影角度 (0° 到 180°)
- $t$：探测器上的位移坐标
- $s$：沿射线方向的积分变量

### 5.2 CPU 实现

CPU后端直接使用 `scikit-image` 的 `radon()` 函数：

```python
sinogram = skimage.transform.radon(image, theta=theta, circle=False)
```

### 5.3 GPU 实现 (CuPy 向量化)

GPU版本使用自定义的双线性插值实现，支持批量角度处理：

**关键步骤：**

1. **探测器大小**：$n_{\text{det}} = \lceil\sqrt{H^2 + W^2}\rceil$（等于图像对角线长度）
2. **坐标映射**：对每个投影角 $\theta$，计算探测器元素 $(t, s)$ 对应的图像坐标 $(x, y)$：

$$
x = t \cos\theta - s \sin\theta + x_c
$$
$$
y = t \sin\theta + s \cos\theta + y_c
$$

1. **双线性插值**：对非整数坐标进行插值采样
2. **沿射线求和**：$\text{sinogram}[\theta, t] = \sum_s f(x, y)$

```python
# 批量处理角度以控制GPU内存
for angle_batch in batches(theta, BATCH_SIZE):
    x_coords = t * cos(θ) - s * sin(θ) + x_center
    y_coords = t * sin(θ) + s * cos(θ) + y_center
    samples = bilinear_interpolate(image, x_coords, y_coords)
    sinogram[:, batch] = samples.sum(axis=s)
```

### 5.4 填充策略

为避免重建时的截断伪影，每个切片在Radon变换前进行两级填充：

1. **方形填充**：非方形切片补零至 $\max(H, W) \times \max(H, W)$
2. **对角线填充**：再补零至 $\lceil\sqrt{2} \times \text{size}\rceil + 32$​，确保旋转时不截断

---

## 6. 滤波反投影重建 (FBP / iRadon)

> **源文件**: `simulation/backends/radon_kernels.py::GPURadonTransform.iradon()`

### 6.1 理论基础

滤波反投影 (Filtered Back Projection, FBP) 是CT重建的经典算法，由 **投影切片定理** 导出：

$$
f(x, y) = \int_0^{\pi} \left[ \mathcal{R}[f](\theta, \cdot) * h(\cdot) \right](x\cos\theta + y\sin\theta) \, d\theta
$$

其中 $h(t)$ 为坡道滤波器 (Ram-Lak/Ramp filter) 的逆傅里叶变换。

### 6.2 算法步骤

#### Step 1: 坡道滤波 (Ramp Filter)

在频域应用坡道滤波器以补偿低频采样过密问题：

$$
H(f) = |f|
$$

```python
# 零填充到2的幂次长度
projection_size_padded = next_power_of_2(2 * n_det)

# 构造坡道滤波器
f = fftfreq(projection_size_padded)
ramp = abs(f)

# 频域滤波
F_sino = FFT(sinogram, axis=detector)
filtered = IFFT(F_sino * ramp, axis=detector).real
```

#### Step 2: 反投影 (Backprojection)

将滤波后的正弦图反投影到图像空间：

对每个输出像素 $(x, y)$，计算其在每个角度 $\theta$ 上的探测器位置：

$$
t = x \cos\theta + y \sin\theta + t_c
$$

然后对滤波后的正弦图进行线性插值采样并求和：

```python
for angle_batch in batches(theta, BATCH_SIZE):
    t_idx = x * cos(θ) + y * sin(θ) + det_center
    value = linear_interpolate(filtered_sinogram, t_idx, angle)
    reconstructed += value

# 最终归一化
reconstructed *= π / (2 * N_angles)
```

### 6.3 CPU vs GPU

| 特性 | CPU (scikit-image) | GPU (CuPy) |
|------|-------------------|-------------|
| 滤波器 | `ramp` (Ram-Lak) | 手动FFT + |f| |
| 插值 | scikit-image 内建 | 手动双线性 |
| 批处理 | 无 | 角度分批(60/batch) |
| 内存 | 主内存 | VRAM, 自动分块 |

---

## 7. 噪声模型 (Poisson 统计)

> **源文件**: `gpu_simulator.py::_simulate_slice_kernel()`, `physical_simulator.py::_simulate_slice()`

### 7.1 物理原理

CT探测器接收到的光子数服从 **Poisson分布**，这是X射线成像中量子噪声的物理来源：

$$
N_{\text{detected}} \sim \text{Poisson}(N_0 \cdot I_{\text{total}})
$$

其中：

- $N_0$：入射光子总数（由管电流mA决定）
- $I_{\text{total}}$：归一化透射强度

### 7.2 光子计数

入射光子数根据管电流线性缩放：

$$
N_0 = N_{\text{base}} \times \frac{I_{\text{tube}}(\text{mA})}{200\text{ mA}}
$$

默认 $N_{\text{base}} = 10^5$ photons/detector element @ 200 mA。

### 7.3 实现策略

```python
I_counts = max(I_total * N_0, 1.0)

if I_counts.max() > 1e7:
    # 高光子计数：用高斯近似避免int64溢出
    # Poisson(λ) ≈ N(λ, √λ)  当 λ >> 1
    I_noisy = Normal(I_counts, sqrt(I_counts))
else:
    # 中等光子计数：使用精确Poisson
    I_noisy = Poisson(I_counts)
```

### 7.4 对数变换 (正弦图)

加噪后的光子计数转换为投影值（正弦图）：

$$
p(\theta, t) = -\ln\left(\frac{I_{\text{noisy}}}{N_0}\right)
$$

这个正弦图随后输入FBP重建。

---

## 8. 散射模拟

> **源文件**: `simulation/physics/scatter.py::ConvolutionScatter`

### 8.1 物理背景

X射线穿过物体时，一部分光子会发生 **Compton散射** 或 **Rayleigh散射**，散射光子到达探测器时偏离原始射线路径，产生：

- **杯状伪影 (Cupping artifact)**：重建图像中心CT值降低
- **对比度降低**：散射信号叠加在初级信号上

### 8.2 卷积散射模型

采用 Siewerdsen & Jaffray (2001) 提出的快速近似方法：

$$
I_{\text{scatter}} = \text{SPR} \times \text{GaussianBlur}(I_{\text{primary}},\, \sigma)
$$

其中：

- **SPR** (Scatter-to-Primary Ratio)：散射-初级比，典型值 0.05–0.30
- $\sigma$：散射核宽度（像素），典型值 30 像素
- 高斯模糊沿探测器方向一维应用（散射主要是横向扩散）

### 8.3 厚度调制

可选的路径长度调制（较厚的物体产生更多散射）：

$$
I_{\text{scatter}} = I_{\text{scatter}} \times \text{clip}\left(\frac{L}{L_{\max}},\, 0.1,\, 1.0\right)
$$

### 8.4 总强度叠加

$$
I_{\text{total}} = I_{\text{primary}} + I_{\text{scatter}}
$$

```python
if scatter_enabled:
    I_scatter = SPR * gaussian_filter1d(I_primary, sigma, axis=detector_axis)
    if path_lengths is not None:
        I_scatter *= clip(path_lengths / max_path, 0.1, 1.0)
    I_total = I_primary + I_scatter
```

### 8.5 参数效果

| SPR | 效果 |
|-----|------|
| 0.05 | 轻微散射（高品质扫描仪） |
| 0.15 | 中等散射（默认值，CBCT典型） |
| 0.30 | 严重散射（大视野cone-beam） |

---

## 9. 运动模糊模拟

> **源文件**: `simulation/physics/scatter.py::MotionBlur`

### 9.1 物理原理

在实际CT扫描中，X射线管和探测器绕患者连续旋转。在单次曝光期间，龙门架会旋转一个小角度（积分角），导致投影数据在角度方向上模糊。

### 9.2 实现

运动模糊建模为正弦图沿 **角度轴** 的一维均匀滤波：

$$
I_{\text{blurred}}(\theta, t) = \frac{1}{K} \sum_{j=0}^{K-1} I_{\text{primary}}(\theta + j\Delta\theta, t)
$$

- 核大小：$K = \text{round}(\theta_{\text{blur}} / \Delta\theta_{\text{step}})$
- 使用 `wrap` 边界模式（旋转连续性）

```python
kernel_size = round(blur_angle_deg / angular_step_deg)
blurred = uniform_filter1d(intensity, size=kernel_size, axis=angular_axis, mode='wrap')
```

### 9.3 参数

| 参数 | 典型值 | 效果 |
|------|--------|------|
| `blur_angle_deg` | 0.5° | 精细扫描，模糊最小 |
| `blur_angle_deg` | 1.0° | 默认，边缘轻微模糊 |
| `blur_angle_deg` | 2.0° | 快速螺旋扫描，明显模糊 |

---

## 10. 相对衰减值归一化

### 10.1 定义

工业CT通常直接使用线性衰减系数或其归一化形式，而不使用医学CT的 Hounsfield 单位。本文采用 **水参照相对衰减值**：

$$
\mu_{\text{rel}} = \frac{\mu_{\text{recon}} - \mu_{\text{water}}^{\text{eff}}}{\mu_{\text{water}}^{\text{eff}}}
$$

其中 $\mu_{\text{water}}^{\text{eff}}$ 使用上述频谱加权有效衰减系数。

### 10.2 参考范围（相对衰减值）

| 材料 | 相对衰减值 $\mu_{\text{rel}}$（典型） |
|------|----------------------------------------|
| 空气 | 约 −1.0 |
| 水 | 0 |
| 铝 | 正值（通常显著高于水） |
| 钛 | 更高正值 |

> 注：工业CT中具体数值与能谱、探测器、重建滤波和束硬化校正强相关，建议以实测标定件建立设备专属标定曲线。

### 10.3 实现

```python
mu_rel_slice = (reconstructed - mu_water_effective) / mu_water_effective
```

---

## 11. 压缩力学仿真

> **源文件**: `simulation/mechanics/elasticity.py`, `simulation/mechanics/manager.py`

### 11.1 概述

压缩仿真模拟物品在外力作用下的形变过程，生成 4D-CT 时间序列。支持两种模式：

| 模式 | 方法 | 精度 | 速度 |
|------|------|------|------|
| 仿射 (Geometric) | 均匀缩放 | 低 | 快 |
| 物理 (Physical) | 弹性力学FDM | 高 | 慢 |

### 11.2 仿射压缩

简单的线性缩放，沿压缩轴按比例压缩：

$$
V'(z, y, x) = V\left(\frac{z}{1 - \epsilon_z},\, y,\, x\right)
$$

使用 `scipy.ndimage.zoom` 实现。

### 11.3 物理压缩 — 线性弹性FDM

#### 11.3.1 问题设定

求解三维线性弹性静力平衡方程：

$$
\nabla \cdot \boldsymbol{\sigma} = 0
$$

边界条件：

- **底面** ($z = 0$)：固定（$\mathbf{u} = 0$）
- **顶面** ($z = N_z - 1$)：施加位移（$u_z = -\epsilon \cdot N_z$）
- **侧面**：自由

#### 11.3.2 刚度映射

从二值体素网格构建刚度场：

$$
k(\mathbf{x}) = \begin{cases}
k_{\text{solid}} = 1.0 & \text{if } V(\mathbf{x}) > 0.5 \\
k_{\text{void}} = 10^{-6} & \text{otherwise}
\end{cases}
$$

#### 11.3.3 多分辨率策略

为降低计算量，使用降采样策略：

1. **降采样**：体积按 `downsample_factor` (默认4×) 缩小
2. **求解**：在粗网格上进行Jacobi迭代
3. **上采样**：位移场三线性插值回原始分辨率
4. **缩放**：$\mathbf{u} \leftarrow \mathbf{u} \times \text{downsample\_factor}$

#### 11.3.4 初始猜测

利用线性弹性解析解作为初始值加速收敛：

- **轴向位移**：线性梯度 $u_z^{(0)} = -\epsilon \cdot z / (N_z - 1) \cdot N_z$
- **横向位移** (Poisson 效应)：$u_x^{(0)} = \nu \cdot \epsilon \cdot x$, $u_y^{(0)} = \nu \cdot \epsilon \cdot y$

#### 11.3.5 Jacobi 松弛迭代

使用加权Jacobi方法求解平衡方程的近似解：

```
repeat N_iterations times:
    u_smooth = convolve3d(u * stiffness, laplacian_kernel) / convolve3d(stiffness, laplacian_kernel)
    u ← (1 − ω) · u + ω · u_smooth
    enforce boundary conditions
    check convergence: max|Δu| < tolerance
```

**6点拉普拉斯核**（3×3×3）：

$$
K = \frac{1}{6}\begin{bmatrix}
\begin{bmatrix}0&0&0\\0&1&0\\0&0&0\end{bmatrix} &
\begin{bmatrix}0&1&0\\1&0&1\\0&1&0\end{bmatrix} &
\begin{bmatrix}0&0&0\\0&1&0\\0&0&0\end{bmatrix}
\end{bmatrix}
$$

- **松弛因子**：$\omega = 0.6$（欠松弛确保稳定性）
- **收敛判据**：$\max|u_z^{(n)} - u_z^{(n-50)}| < 10^{-4}$
- **典型迭代次数**：300–500

#### 11.3.6 位移场应用

使用 `map_coordinates` 进行反向映射变形：

$$
V'(\mathbf{x}) = V(\mathbf{x} - \mathbf{u}(\mathbf{x}))
$$

GPU版本使用分块处理避免OOM：

```python
for chunk in split(z_range, chunk_size=64):
    coords = stack([z - u_z, y - u_y, x - u_x])   # 变形坐标
    warped[chunk] = map_coordinates(volume, coords)  # 三线性插值
```

### 11.4 多步压缩

`CompressionManager` 将总压缩量均匀分配到多个步骤：

$$
\epsilon_{\text{step}} = \frac{\epsilon_{\text{total}}}{N_{\text{steps}}} \times \text{step}
$$

每步产生一个 `CompressionResult`，包含变形体积、体素尺寸和可选注释。

---

## 12. GPU 加速策略

> **源文件**: `simulation/physics/gpu_simulator.py`, `simulation/backends/radon_kernels.py`

### 12.1 整体GPU流水线

```
┌─ Transfer Stream ─────────────────────┐
│  batch_data ──→ GPU                   │
│       ↓ (event sync)                  │
│  ┌─ Compute Stream ────────────────┐  │
│  │  for each slice in batch:       │  │
│  │    1. Pad to diagonal           │  │
│  │    2. GPU Radon (batched angles) │  │
│  │    3. Polychromatic attenuation  │  │
│  │    4. Motion blur (optional)     │  │
│  │    5. Scatter (optional)         │  │
│  │    6. Poisson noise              │  │
│  │    7. -log transform             │  │
│  │    8. GPU iRadon (batched)       │  │
│  │    9. Crop & attenuation normalize│  │
│  └──────────────────────────────────┘  │
│       ↓ (async transfer back)          │
│  result ──→ CPU                        │
└────────────────────────────────────────┘
```

### 12.2 自动内存管理

GPU模拟器在初始化时自动检测可用VRAM，并动态调整批大小：

```python
available = free_vram * 0.6   # 60%安全裕量

slice_batch  = clamp(available / (size² * 4 * 80), 1, 16)
radon_batch  = clamp(available / (n_det² * 24), 5, 60)
iradon_batch = clamp(available / (size² * 16), 10, 120)
```

### 12.3 双流异步流水线

使用两个CUDA流实现计算-传输重叠：

1. **Transfer Stream**：异步上传下一批数据
2. **Compute Stream**：处理当前批次
3. **Event同步**：确保数据就绪后才开始计算

### 12.4 性能关键优化

| 优化 | 技术 | 效果 |
|------|------|------|
| 向量化Radon | 批量角度并行 | ~20× vs 逐角度 |
| 原位 (in-place) 操作 | `+=` 累加避免分配 | 减少显存碎片 |
| 及时释放 | `del` + 手动池清理 | 避免OOM |
| 高斯噪声近似 | $N_0 > 10^7$ 时用正态分布 | 避免int64溢出 |
| 分块弹性变形 | 64-slice chunks | 大体积不OOM |

---

## 13. 简化模式 vs 物理模式对比

| 特性 | SimpleCTSimulator | PhysicalCTSimulator |
|------|-------------------|---------------------|
| 能谱 | 单能 | 多色 (Kramers + 特征辐射) |
| 衰减 | 经验映射近似 | NIST能量相关μ(E) |
| 噪声 | 可选高斯 | Poisson物理统计 |
| 散射 | 无 | 卷积近似 |
| 运动模糊 | 无 | 角度积分 |
| 射束硬化 | 无 | 自然产生 |
| 定量一致性 | 近似线性映射 | 物理推导 |
| 速度 | 快 (~秒级) | 较慢 (~分钟级) |
| GPU支持 | ✓ | ✓ (双流流水线) |
| 用途 | 快速预览 | AI训练数据生成 |

### 简化模式流程

```python
# SimpleCTSimulator 仅做:
sinogram = radon(slice)        # 无衰减系数
recon = iradon(sinogram)       # 直接FBP
mu_rel = normalize_to_relative_mu(recon)  # 线性映射到相对衰减范围
```

### 物理模式流程

```python
# PhysicalCTSimulator 完整流程:
spectrum = generate_spectrum(kVp, filtration)
mu_obj = NIST_lookup(material, bin_energies)
mu_bg = NIST_lookup(background, bin_energies)

for each slice:
    path = radon(slice)
    I = Σ w[i] * exp(-mu_obj[i]*L_obj - mu_bg[i]*L_bg)  # 多色
    I = motion_blur(I)                                     # 可选
    I = I + scatter(I)                                      # 可选
    I_noisy = poisson(I * N_photons)                        # 量子噪声
    sinogram = -log(I_noisy / N_photons)
    recon = iradon(sinogram)
    mu_rel = (recon - mu_water) / mu_water                   # 相对衰减值
```

---

## 14. Future Works：提升物理模拟准度

以下条目按“对最终定量值 / 纹理真实性 / 几何保真度的潜在增益”进行优先排序，重点聚焦 **accuracy first**。

### 14.1 高优先级（建议优先实现）

| 方向 | 现状 | 建议方法 | 预期收益 |
|------|------|----------|----------|
| 多色前向模型 | 当前使用能量分箱（默认10 bins） | 引入**自适应能量分箱**（按谱梯度和材料K-edge邻域加密），并对厚路径区域使用更细分箱 | 降低 beam hardening 近似误差，改善高衰减材料定量偏差 |
| 探测器物理响应 | 当前主要建模光子统计噪声 | 增加**探测器响应链**：闪烁体吸收效率、光电转换增益、电子读出噪声、暗电流偏置、像素间串扰（PSF/MTF） | 重建纹理与噪声功率谱更接近真实设备，提升域一致性 |
| 散射模型 | 当前卷积近似散射 | 采用**混合散射建模**：低频分量用核卷积，高衰减区域用稀疏 Monte Carlo 校正；按角度/厚度自适应融合 | 降低散射欠补偿/过补偿，提升低对比区域定量准确度 |
| 几何标定误差 | 当前默认理想几何 | 引入**系统几何参数扰动模型**（源-探测器距离、中心偏移、像素尺寸误差、角度抖动）并支持标定参数回归 | 减少条纹/模糊伪影，提升分辨率与边缘定位准确度 |

### 14.2 中优先级（中期迭代）

| 方向 | 建议方法 | 预期收益 |
|------|----------|----------|
| 体素部分容积效应 | 在 Radon 前加入**超采样/抗锯齿投影**（sub-voxel occupancy 或 distance transform based line integral） | 缓解小缺陷边缘“台阶化”，提高微小孔洞可检出性 |
| 运动模型真实性 | 将当前角度积分模糊扩展为**时序运动场模型**（非刚体+呼吸/机械周期），与投影角同步采样 | 动态伪影更真实，提升 4D 场景训练有效性 |
| 衰减标定链 | 增加**多材料 phantom 标定流程**：水/空气/骨/金属基准，拟合非线性校正曲线（分 kVp） | 绝对衰减值偏差可控，跨扫描参数的一致性更高 |
| 重建算法 | 在 FBP 外新增**多色一致性迭代重建**（PWLS/MBIR-lite）作为高精模式 | 降低噪声与条纹伪影，提高定量重建稳定性 |

### 14.3 压缩耦合（4D 可信度提升）

当前 4D 标注的中心/尺度更新以全局缩放为主。为提高“力学-成像”一致性，建议：

1. 使用**位移场驱动的逐缺陷变换**（直接采样 $\nabla \mathbf{u}$）替代全局平均缩放。
2. 引入**各向异性和非线性材料模型**（如 neo-Hookean / 分段线弹性）以覆盖大形变。
3. 对每一时间步输出**不确定度标签**（位置方差、体积置信区间），用于训练时的损失加权。

这将显著提升时序标注与真实变形的一致性，尤其在高压缩比和复杂结构下。

### 14.4 可量化评估指标（建议纳入CI/实验报告）

为避免“只看视觉效果”，建议将以下指标作为 future work 的验收标准：

- **相对衰减定量误差**：$|\mu_{rel,sim} - \mu_{rel,ref}|$（按材料分组统计 MAE / P95）
- **噪声统计一致性**：NPS（Noise Power Spectrum）与真实扫描对齐度
- **分辨率指标**：MTF50 / MTF10
- **伪影强度**：beam hardening cup artifact 指标、金属伪影条纹幅值
- **下游任务收益**：同一模型在真实数据集上的检测/分割性能增益（AUC / mAP / Dice）

### 14.5 建议实施路线（Accuracy-first）

1. **Phase A（短期）**：自适应能量分箱 + 多材料衰减标定 + 几何扰动建模
2. **Phase B（中期）**：探测器响应链 + 混合散射模型 + 部分容积超采样
3. **Phase C（中长期）**：多色一致性迭代重建 + 位移场驱动4D标注 + 非线性力学

---

## 15. 参考文献

1. **Kramers, H.A.** (1923). "On the theory of X-ray absorption and of the continuous X-ray spectrum." *Philosophical Magazine*, 46(275), 836-871.

2. **NIST XCOM Database**. Photon Cross Sections Database. <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>

3. **Kak, A.C. & Slaney, M.** (1988). *Principles of Computerized Tomographic Imaging*. IEEE Press. — FBP算法的经典教材。

4. **Siewerdsen, J.H. & Jaffray, D.A.** (2001). "Cone-beam computed tomography with a flat-panel imager: Magnitude and effects of x-ray scatter." *Medical Physics*, 28(2), 220-231. — 卷积散射模型参考。

5. **Buzug, T.M.** (2008). *Computed Tomography: From Photon Statistics to Modern Cone-Beam CT*. Springer. — CT噪声模型综述。

6. **Hubbell, J.H. & Seltzer, S.M.** (1995). "Tables of X-Ray Mass Attenuation Coefficients and Mass Energy-Absorption Coefficients." *NISTIR 5632*. — 质量衰减系数数据来源。

---

> **本文档由 CT Simulation 项目自动维护。最后更新：2026-02-13。**
