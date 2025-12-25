# 🔫 PUBG Intelligent Tactical Terminal (Pro)
### 基于 PyTorch 深度学习的绝地求生武器管理与战场态势感知系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/Course-Project-green)

## 📖 项目简介 (Introduction)

本项目是一个集成了**武器资产管理**与**AI 战场音频分析**的综合战术终端。
系统采用 **PyTorch** 深度学习框架，构建了双流融合网络，实现了对战场枪声的**型号识别**（准确率 >88%）和**距离估算**（误差 <30m），并提供了 **CLI (命令行)** 和 **Web (网页端)** 两种交互方式。

---

## ✨ 核心功能 (Features)

### 🎒 模块一：武器资产管理 (Level A)
- **CRUD 操作**：制造、查看、修改、丢弃武器。
- **数据持久化**：使用 JSON/CSV 存储用户档案与全局武器库。
- **高级功能**：
    - 支持按伤害值排序（升序/降序）。
    - 支持模糊搜索武器名称。
    - 武器属性包含：伤害、射击模式、弹夹容量、备弹数、膛内子弹。

### 🎧 模块二：AI 态势感知 (Level B - Advanced)
- **Task 1: 枪声分类 (Classification)**
    - **算法**：1D-CNN (卷积神经网络)。
    - **特征**：MFCC + Delta MFCC + Spectral Contrast。
    - **策略**：Class Weights 类别权重惩罚，解决样本不平衡。
- **Task 2: 距离测算 (Regression/Classification)**
    - **算法**：**Fusion-CRNN (双流融合网络)**。
    - **架构**：Log-Mel 频谱图 (CNN流) + 统计特征 (统计流) -> LSTM 时序分析 -> Attention 机制。
    - **能力**：精准识别枪声距离，并给出战术建议（贴脸/中距离/远距离）。

### 🔐 模块三：权限与安全
- **双重身份**：普通指挥官 (User) vs 基地管理员 (Admin)。
- **安全验证**：SHA-256 密码加密存储。
- **输入遮蔽**：CLI 模式下密码输入显示为 `*` 号。

---

## 📂 项目结构 (Structure)

```text
E:\PUBG_new\
│  app.py                   # [Web入口] Streamlit 图形化界面
│  main.py                  # [CLI入口] 命令行交互界面
│  requirements.txt         # [依赖] 项目依赖库列表
│  README.md                # [文档] 项目说明书
│
├─ data
│      Arms.csv             # 原始武器数据
│      players.json         # 用户存档数据
│      admins.json          # 管理员账户数据
│
├─ core                     # 核心业务逻辑
│      player.py            # 玩家类
│      weapon.py            # 武器类
│      storage.py           # 数据读写接口
│
├─ ai_modules               # 人工智能模块
│      feature_extractor.py # 特征提取器
│      cnn_model.py         # 枪声分类模型架构
│      distance_model.py    # 距离融合模型架构
│      model_trainer.py     # 分类模型训练脚本
│      distance_trainer.py  # 距离模型训练脚本
│      sound_inference.py   # 推理接口
│
└─ models                   # 训练好的模型文件
       cnn_audio_model.pth
       fusion_distance_model.pth
       ...
