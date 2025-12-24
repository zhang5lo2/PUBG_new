import streamlit as st
import pandas as pd
import os
import joblib
import torch
import numpy as np
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# === 导入后端核心模块 ===
from player import Player
from weapon import Weapon
from storage import load_players, save_players, load_global_weapon_library, load_admins
from feature_extractor import extract_features
from cnn_model import GunshotCNN
# 注意：这里我们直接调用 model_trainer 的函数，或者在前端重写部分逻辑以适配 UI
from model_trainer import train_audio_model

# === 页面配置 (商业化外观) ===
st.set_page_config(
    page_title="PUBG 智能战术终端",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 路径配置 ===
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_audio_model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 样式美化 (CSS) ===
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 { color: #FF9900; }
</style>
""", unsafe_allow_html=True)

# === 会话状态初始化 ===
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'players_data' not in st.session_state:
    st.session_state['players_data'] = load_players()
if 'global_weapons' not in st.session_state:
    st.session_state['global_weapons'] = load_global_weapon_library()

# === 工具函数 ===
def save_current_state():
    """保存当前所有玩家数据"""
    save_players(st.session_state['players_data'])

def hash_pwd(password):
    return hashlib.sha256(password.encode()).hexdigest()

# === 侧边栏：登录/注册 ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/PUBG_Logo.svg/1200px-PUBG_Logo.svg.png", width=200)
    st.title("🪖 指挥官控制台")
    
    if st.session_state['user'] is None:
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            login_sid = st.text_input("学号 (Student ID)", key="l_sid")
            login_pwd = st.text_input("密码 (Password)", type="password", key="l_pwd")
            if st.button("🚀 登录系统"):
                players = st.session_state['players_data']
                if login_sid in players and players[login_sid].check_password(login_pwd):
                    st.session_state['user'] = players[login_sid]
                    st.success("身份验证通过！")
                    st.rerun()
                else:
                    st.error("账号或密码错误")
        
        with tab2:
            reg_sid = st.text_input("新学号", key="r_sid")
            reg_pwd1 = st.text_input("设置密码", type="password", key="r_pwd1")
            reg_pwd2 = st.text_input("确认密码", type="password", key="r_pwd2")
            if st.button("📝 注册入伍"):
                players = st.session_state['players_data']
                if reg_sid in players:
                    st.error("该学号已存在")
                elif reg_pwd1 != reg_pwd2:
                    st.error("两次密码不一致")
                elif not reg_sid or not reg_pwd1:
                    st.error("信息不能为空")
                else:
                    new_p = Player(reg_sid)
                    new_p.set_password(reg_pwd1)
                    # 自动发枪
                    gw = st.session_state['global_weapons']
                    if gw:
                        for w in gw.values(): new_p.add_weapon(w)
                    players[reg_sid] = new_p
                    save_current_state()
                    st.success("注册成功！请登录。")
    else:
        # 已登录状态
        user = st.session_state['user']
        st.write(f"👤 **指挥官: {user.student_id}**")
        st.write(f"🎒 武器数量: {len(user.weapons)}")
        
        if st.button("🚪 退出登录"):
            st.session_state['user'] = None
            st.rerun()
            
        st.markdown("---")
        if st.button("💀 注销账号 (慎用)", type="primary"):
             check_pwd = st.text_input("输入密码确认注销", type="password")
             if check_pwd:
                 if user.check_password(check_pwd):
                     del st.session_state['players_data'][user.student_id]
                     save_current_state()
                     st.session_state['user'] = None
                     st.warning("账号已注销")
                     st.rerun()
                 else:
                     st.error("密码错误")

# === 主界面逻辑 ===
if st.session_state['user']:
    user = st.session_state['user']
    
    # 顶部导航 Tabs
    menu = st.tabs(["🎒 武器库管理", "🎤 AI 枪声识别", "🔐 管理员后台"])
    
    # --- Tab 1: 武器库管理 ---
    with menu[0]:
        st.header("📦 战术背包概览")
        
        # 将武器转为 DataFrame 用于展示
        if user.weapons:
            w_data = []
            for i, w in enumerate(user.weapons):
                w_data.append({
                    "序号": i+1,
                    "名称": w.name,
                    "伤害": w.damage,
                    "模式": w.firing_mode,
                    "枪膛/弹夹": f"{w.current_ammo}/{w.magazine_capacity}",
                    "备弹数": w.reserve_mags,
                    "总备弹": w.reserve_mags * w.magazine_capacity
                })
            df = pd.DataFrame(w_data)
            st.dataframe(df, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总火力 (DPS)", sum([w.damage for w in user.weapons]))
            with col2:
                st.metric("总弹药储备", sum([w.total_ammo for w in user.weapons]))
        else:
            st.info("背包空空如也...")

        st.markdown("---")
        st.subheader("🛠️ 战术操作")
        
        c1, c2, c3 = st.columns(3)
        
        # 1. 制造武器
        with c1:
            with st.expander("➕ 制造新武器"):
                n_name = st.text_input("名称", "M416-Custom")
                n_dmg = st.number_input("伤害", 0, 200, 40)
                n_mode = st.text_input("模式", "Auto")
                n_cap = st.number_input("容量", 1, 100, 30)
                n_res = st.number_input("备弹", 0, 20, 3)
                n_cur = st.number_input("枪膛", 0, 100, 30)
                
                if st.button("立即制造"):
                    new_w = Weapon(n_name, n_dmg, n_mode, n_cap, n_res, n_cur)
                    user.add_weapon(new_w)
                    save_current_state()
                    st.success(f"{n_name} 已入库")
                    st.rerun()

        # 2. 改装武器
        with c2:
            with st.expander("🔧 改装/维护"):
                if user.weapons:
                    w_opts = [f"{i+1}. {w.name}" for i, w in enumerate(user.weapons)]
                    sel_w_str = st.selectbox("选择武器", w_opts)
                    idx = int(sel_w_str.split('.')[0]) - 1
                    target_w = user.weapons[idx]
                    
                    st.caption(f"当前改装: {target_w.name}")
                    new_w_name = st.text_input("重命名", target_w.name)
                    new_w_cur = st.slider("调整枪膛子弹", 0, target_w.magazine_capacity, target_w.current_ammo)
                    new_w_res = st.number_input("调整备弹数", 0, 50, target_w.reserve_mags)
                    
                    if st.button("确认改装"):
                        target_w.name = new_w_name
                        target_w.current_ammo = new_w_cur
                        target_w.reserve_mags = new_w_res
                        save_current_state()
                        st.success("改装完成！")
                        st.rerun()
                else:
                    st.write("无武器可改")

        # 3. 排序与丢弃
        with c3:
            with st.expander("🗑️ / 📊 管理"):
                if user.weapons:
                    # 排序
                    sort_order = st.radio("排序方式", ["伤害降序 (强->弱)", "伤害升序 (弱->强)"])
                    if st.button("执行排序"):
                        user.sort_weapons_by_damage(reverse=(sort_order == "伤害降序 (强->弱)"))
                        save_current_state()
                        st.rerun()
                    
                    st.divider()
                    # 丢弃
                    del_opts = [f"{i+1}. {w.name}" for i, w in enumerate(user.weapons)]
                    sel_del = st.selectbox("选择丢弃", del_opts, key="del_sel")
                    if st.button("🗑️ 确认丢弃", type="primary"):
                        idx = int(sel_del.split('.')[0]) - 1
                        user.remove_weapon_by_index(idx)
                        save_current_state()
                        st.warning("武器已销毁")
                        st.rerun()

    # --- Tab 2: AI 枪声识别 ---
    with menu[1]:
        st.header("🎧 战术听音辨位系统 (AI)")
        st.info("基于 PyTorch CNN 深度学习内核 | 支持 MFCC+Delta 特征分析")
        
        uploaded_file = st.file_uploader("上传 MP3 录音文件", type=["mp3"])
        
        if uploaded_file is not None:
            # 保存临时文件
            temp_path = BASE_DIR / "temp_upload.mp3"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio(uploaded_file)
            
            if st.button("🔍 开始识别"):
                if not MODEL_PATH.exists():
                    st.error("❌ AI 模型未加载！请联系管理员进行训练。")
                else:
                    with st.spinner("正在提取声学特征 (MFCC, Spectral, Chroma)..."):
                        # 调用后端逻辑
                        features = extract_features(temp_path)
                    
                    if features is not None:
                        try:
                            # 加载模型
                            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
                            scaler = joblib.load(SCALER_PATH)
                            
                            # 预处理
                            features = features.reshape(1, -1)
                            features_scaled = scaler.transform(features)
                            features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
                            
                            # 模型推理
                            model = GunshotCNN(checkpoint['num_features'], checkpoint['num_classes']).to(DEVICE)
                            model.load_state_dict(checkpoint['state_dict'])
                            model.eval()
                            
                            with torch.no_grad():
                                outputs = model(features_tensor)
                                probs = torch.nn.functional.softmax(outputs, dim=1)
                                confidence, pred_idx = torch.max(probs, 1)
                                
                                label = checkpoint['classes'][pred_idx.item()]
                                conf_val = confidence.item() * 100
                            
                            # 结果展示
                            st.success(f"🎯 识别结果: **{label.upper()}**")
                            st.progress(int(conf_val))
                            st.caption(f"置信度: {conf_val:.2f}%")
                            
                            # 显示 Top-3 概率
                            st.write("📊 概率分布 (Top 3):")
                            top3_prob, top3_idx = torch.topk(probs, 3)
                            for i in range(3):
                                p = top3_prob[0][i].item() * 100
                                l = checkpoint['classes'][top3_idx[0][i].item()]
                                st.write(f"- {l.upper()}: {p:.1f}%")
                                
                        except Exception as e:
                            st.error(f"推理失败: {e}")
                    else:
                        st.error("特征提取失败，请检查音频文件。")
            
            # 清理临时文件
            # if temp_path.exists(): os.remove(temp_path)

    # --- Tab 3: 管理员后台 ---
    with menu[2]:
        st.header("🔐 基地指挥中心")
        
        if 'admin_logged_in' not in st.session_state:
            st.session_state['admin_logged_in'] = False
            
        if not st.session_state['admin_logged_in']:
            admin_user = st.text_input("管理员账号", key="ad_user")
            admin_pass = st.text_input("管理员密码", type="password", key="ad_pass")
            
            if st.button("验证权限"):
                admins = load_admins()
                if admin_user in admins:
                    if hashlib.sha256(admin_pass.encode()).hexdigest() == admins[admin_user]:
                        st.session_state['admin_logged_in'] = True
                        st.success("权限已解锁")
                        st.rerun()
                    else:
                        st.error("密码错误")
                else:
                    st.error("账号不存在")
        else:
            st.success("✅ 管理员权限已激活")
            if st.button("🔒 锁定终端"):
                st.session_state['admin_logged_in'] = False
                st.rerun()
            
            st.divider()
            st.subheader("🤖 AI 模型训练")
            st.warning("⚠️ 训练过程可能占用大量计算资源，请勿关闭页面。")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.info("训练策略: 物理音频增强 + 类别权重惩罚")
            with col_t2:
                st.info(f"设备: {DEVICE}")
                
            if st.button("🚀 启动深度学习训练 (PyTorch)", type="primary"):
                # 这里我们利用 streamlit 的特性，捕获控制台输出比较麻烦
                # 建议直接调用函数，并在完成后显示结果
                progress_text = "正在扫描音频、增强数据并训练神经网络，请稍候..."
                my_bar = st.progress(0, text=progress_text)
                
                try:
                    # 这是一个耗时操作
                    with st.spinner("AI 正在学习中 (这可能需要几分钟)..."):
                        # 直接调用之前的 model_trainer 逻辑
                        # 为了能在前端看到反馈，最好去修改 model_trainer 返回一些信息
                        # 但这里为了兼容，直接调用，看控制台日志即可
                        train_audio_model()
                    
                    my_bar.progress(100, text="训练完成！")
                    st.success("✅ 模型训练完毕并已保存！")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"训练出错: {e}")

else:
    # 未登录首页展示
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🔫 PUBG 智能战术终端</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Elite Weapon Management System</h3>", unsafe_allow_html=True)
    st.info("请在左侧侧边栏登录或注册以开始使用。")
