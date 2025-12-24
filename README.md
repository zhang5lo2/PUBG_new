# 🔫 PUBG 智能战术终端 (AI Gunshot Recognition)

基于 PyTorch 深度学习与 Streamlit 的绝地求生武器管理与枪声识别系统。

## 🌟 功能特性
- **武器库管理**: 装备自定义、改装、排序。
- **AI 枪声识别**: 上传 MP3，识别武器类型（支持 38 种武器）。
- **硬核内核**: 1D-CNN + MFCC/Delta/Contrast 特征提取 + 物理数据增强。
- **权限分级**: 普通指挥官 vs 基地管理员。

## 🚀 快速启动
1. 安装依赖: `pip install -r requirements.txt`
2. 启动系统: `streamlit run app.py`

## 📊 模型表现
- Micro-F1: ~0.86
- Macro-F1: ~0.82
