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

# === 导入 AI 相关模块 ===
from feature_extractor import extract_features
from cnn_model import GunshotCNN
from model_trainer import train_audio_model

# === 导入距离识别模块 (Fusion) ===
from distance_model import FusionCRNN
from distance_trainer import train_distance_model, extract_dual_features

# === 页面配置 ===
st.set_page_config(
    page_title="PUBG 智能战术终端 Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 路径配置 ===
BASE_DIR = Path(__file__).resolve().parent

# 1. 分类模型文件
CLS_MODEL_PATH = BASE_DIR / "cnn_audio_model.pth"
CLS_SCALER_PATH = BASE_DIR / "scaler.pkl"

# 2. 距离模型文件 (FusionCRNN + Classification)
DIST_MODEL_PATH = BASE_DIR / "fusion_distance_model.pth"
DIST_SCALER_PATH = BASE_DIR / "fusion_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CSS 美化 ===
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #d35400; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        border-radius: 8px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;
    }
    /* 进度条颜色 */
    .stProgress > div > div > div > div {
        background-color: #f1c40f;
    }
</style>
""", unsafe_allow_html=True)

# === 会话状态初始化 ===
if 'user' not in st.session_state: st.session_state['user'] = None
if 'players_data' not in st.session_state: st.session_state['players_data'] = load_players()
if 'global_weapons' not in st.session_state: st.session_state['global_weapons'] = load_global_weapon_library()

# === 辅助函数 ===
def save_state():
    save_players(st.session_state['players_data'])

# === 侧边栏：用户系统 ===
with st.sidebar:
    st.title("🪖 指挥官控制台")
    st.markdown("---")
    
    if st.session_state['user'] is None:
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            sid = st.text_input("学号", key="l_sid")
            pwd = st.text_input("密码", type="password", key="l_pwd")
            if st.button("🚀 进入系统", type="primary"):
                players = st.session_state['players_data']
                if sid in players and players[sid].check_password(pwd):
                    st.session_state['user'] = players[sid]
                    st.toast("欢迎回来，指挥官！", icon="🫡")
                    st.rerun()
                else: st.error("验证失败")
        with tab2:
            r_sid = st.text_input("新学号", key="r_sid")
            r_pwd = st.text_input("设置密码", type="password", key="r_pwd")
            if st.button("📝 注册"):
                players = st.session_state['players_data']
                if r_sid in players: st.error("已存在")
                elif not r_sid or not r_pwd: st.error("不能为空")
                else:
                    new_p = Player(r_sid)
                    new_p.set_password(r_pwd)
                    gw = st.session_state['global_weapons']
                    if gw:
                        for w in gw.values(): new_p.add_weapon(w)
                    players[r_sid] = new_p
                    save_state()
                    st.success("注册成功")
    else:
        user = st.session_state['user']
        st.write(f"👤 **{user.student_id}**")
        st.caption("状态: 在线 | 权限: 指挥官")
        if st.button("🚪 退出"):
            st.session_state['user'] = None
            st.rerun()
        st.divider()
        with st.expander("💀 危险区域"):
            if st.button("注销账号", type="primary"):
                pwd = st.text_input("确认密码", type="password")
                if pwd and user.check_password(pwd):
                    del st.session_state['players_data'][user.student_id]
                    save_state()
                    st.session_state['user'] = None
                    st.rerun()

# === 主界面 ===
if st.session_state['user']:
    user = st.session_state['user']
    
    # 三大功能模块
    tabs = st.tabs(["🎒 武器库 (Inventory)", "🎧 战场态势感知 (AI Analysis)", "🔐 基地后台 (Admin)"])
    
    # --- Tab 1: 武器管理 ---
    with tabs[0]:
        c1, c2 = st.columns([3, 1])
        with c1: st.subheader("📦 战术背包管理")
        with c2: 
            search_query = st.text_input("🔍 搜索武器...", placeholder="输入名称 (如 AK)").strip().lower()
        
        display_weapons = user.weapons
        if search_query:
            display_weapons = [w for w in user.weapons if search_query in w.name.lower()]
            if not display_weapons:
                st.warning(f"🚫 未找到包含 '{search_query}' 的武器")
        
        if display_weapons:
            # 数据展示
            data = []
            for i, w in enumerate(display_weapons):
                data.append({
                    "名称": w.name, "伤害": w.damage, "模式": w.firing_mode,
                    "弹药状态": f"{w.current_ammo} / {w.reserve_mags * w.magazine_capacity}",
                    "备弹数": w.reserve_mags
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            
            # 统计面板
            m1, m2, m3 = st.columns(3)
            m1.metric("显示数量", len(display_weapons))
            m2.metric("平均伤害", int(sum(w.damage for w in display_weapons)/len(display_weapons)) if len(display_weapons)>0 else 0)
            m3.metric("总弹药储备", sum(w.total_ammo for w in display_weapons))
        else:
            if not search_query: st.info("背包是空的，快去制造武器吧！")

        st.divider()
        
        # 操作区
        col_op1, col_op2 = st.columns(2)
        with col_op1:
            with st.expander("🛠️ 制造/改装"):
                tab_mk, tab_mod = st.tabs(["制造", "改装"])
                with tab_mk:
                    n_name = st.text_input("名称", "M416-Custom")
                    cols = st.columns(3)
                    dmg = cols[0].number_input("伤害", 0, 999, 40)
                    cap = cols[1].number_input("容量", 1, 200, 30)
                    cur = cols[2].number_input("膛内", 0, 200, 30)
                    if st.button("制造"):
                        user.add_weapon(Weapon(n_name, dmg, "Auto", cap, 5, cur))
                        save_state()
                        st.success("制造完成")
                        st.rerun()
                with tab_mod:
                    if user.weapons:
                        sel_idx = st.selectbox("选择武器", range(len(user.weapons)), format_func=lambda x: user.weapons[x].name)
                        w = user.weapons[sel_idx]
                        new_name = st.text_input("改名", w.name)
                        new_res = st.slider("备弹数", 0, 20, w.reserve_mags)
                        if st.button("应用改装"):
                            w.name = new_name
                            w.reserve_mags = new_res
                            save_state()
                            st.rerun()
        
        with col_op2:
             with st.expander("⚖️ 排序/丢弃"):
                 c_sort1, c_sort2 = st.columns(2)
                 if c_sort1.button("📉 伤害降序"):
                     user.sort_weapons_by_damage(True)
                     save_state()
                     st.rerun()
                 if c_sort2.button("📈 伤害升序"):
                     user.sort_weapons_by_damage(False)
                     save_state()
                     st.rerun()
                 
                 st.divider()
                 del_idx = st.selectbox("选择丢弃对象", range(len(user.weapons)), format_func=lambda x: user.weapons[x].name, key="del_s")
                 if st.button("🗑️ 确认丢弃", type="primary"):
                     user.remove_weapon_by_index(del_idx)
                     save_state()
                     st.toast("武器已销毁")
                     st.rerun()

    # --- Tab 2: AI 态势感知 (Fusion 双模) ---
    with tabs[1]:
        st.header("🎧 战场声学分析终端")
        st.markdown("""
        > **系统内核**: `PyTorch 2.6`  
        > **分类引擎**: `GunshotNet (1D-CNN)` - 识别武器型号  
        > **测距引擎**: `FusionCRNN (Dual-Stream)` - 多模态融合测距
        """)
        
        upload = st.file_uploader("上传战场录音 (.mp3)", type=["mp3"])
        
        if upload:
            tpath = BASE_DIR / "temp_analyzing.mp3"
            with open(tpath, "wb") as f: f.write(upload.getbuffer())
            st.audio(upload)
            
            if st.button("🚀 启动全方位分析"):
                if not CLS_MODEL_PATH.exists() or not DIST_MODEL_PATH.exists():
                    st.error("❌ 模型缺失，请联系管理员训练模型！")
                else:
                    status_container = st.status("正在进行声学特征解算...", expanded=True)
                    
                    try:
                        # === Task 1: 枪声分类 ===
                        status_container.write("Task 1: 提取 MFCC & Delta 特征...")
                        feats = extract_features(tpath)
                        
                        label_res = "Unknown"
                        conf_res = 0.0
                        
                        if feats is not None:
                            ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE, weights_only=False)
                            scaler = joblib.load(CLS_SCALER_PATH)
                            
                            inp = torch.FloatTensor(scaler.transform(feats.reshape(1, -1))).to(DEVICE)
                            model_cls = GunshotCNN(ckpt['num_features'], ckpt['num_classes']).to(DEVICE)
                            model_cls.load_state_dict(ckpt['state_dict'])
                            model_cls.eval()
                            
                            with torch.no_grad():
                                out = model_cls(inp)
                                prob = torch.nn.functional.softmax(out, dim=1)
                                conf, idx = torch.max(prob, 1)
                                label_res = ckpt['classes'][idx.item()]
                                conf_res = conf.item() * 100
                        
                        # === Task 2: 距离分类 (FusionCRNN) ===
                        status_container.write("Task 2: Log-Mel + Stat 双流融合分析...")
                        
                        # 提取双流特征
                        m, s = extract_dual_features(tpath, augment=False)
                        dist_res = "Unknown"
                        
                        if m is not None and s is not None:
                            # 归一化 Stat
                            ss = joblib.load(DIST_SCALER_PATH)
                            sn = ss.transform(s.T).T
                            
                            # 转 Tensor
                            tm = torch.FloatTensor(m).unsqueeze(0).unsqueeze(0).to(DEVICE)
                            ts = torch.FloatTensor(sn).unsqueeze(0).to(DEVICE)
                            
                            # 加载模型
                            dp = torch.load(DIST_MODEL_PATH, map_location=DEVICE, weights_only=False)
                            md = FusionCRNN(dp['num_classes']).to(DEVICE)
                            md.load_state_dict(dp['state_dict'])
                            md.eval()
                            
                            with torch.no_grad():
                                out = md(tm, ts)
                                p = torch.nn.functional.softmax(out, dim=1)
                                _, i = torch.max(p, 1)
                                dist_res = dp['classes'][i.item()]
                        
                        status_container.update(label="分析完成", state="complete", expanded=False)
                        
                        # === 结果展示看板 ===
                        st.markdown("### 📊 战术情报")
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.info(f"🔫 武器判定: **{label_res.upper()}**")
                            st.progress(int(conf_res), text=f"AI 置信度: {conf_res:.1f}%")
                            
                        with res_col2:
                            st.warning(f"📏 距离判定: **{dist_res}**")
                            
                            # 战术建议
                            d_str = str(dist_res)
                            if "0m" in d_str or "10m" in d_str or "20m" in d_str:
                                st.error("🚨 战术建议: 极度危险！(CQC)")
                            elif "50m" in d_str or "100m" in d_str:
                                st.warning("⚠️ 战术建议: 中距离接触 (Mid-Range)")
                            else:
                                st.success("✅ 战术建议: 远距离目标 (Long-Range)")
                                
                    except Exception as e:
                        st.error(f"分析过程中发生错误: {e}")
                        # st.exception(e) # 调试时可打开

    # --- Tab 3: 管理员后台 ---
    with tabs[2]:
        st.header("🔐 核心训练后台")
        
        if 'is_admin' not in st.session_state: st.session_state['is_admin'] = False
        
        if not st.session_state['is_admin']:
            with st.form("admin_login"):
                ad_user = st.text_input("Admin ID")
                ad_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Verify Access"):
                    admins = load_admins()
                    if admins and ad_user in admins and hashlib.sha256(ad_pass.encode()).hexdigest() == admins[ad_user]:
                        st.session_state['is_admin'] = True
                        st.rerun()
                    else: st.error("Access Denied")
        else:
            st.success("管理员权限已授予 (Root Access Granted)")
            if st.button("🔒 Lock Terminal"):
                st.session_state['is_admin'] = False
                st.rerun()
            
            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("1. 枪声分类模型")
                st.caption("架构: 1D-CNN | 策略: Class Weights")
                if st.button("🚀 启动分类训练"):
                    with st.spinner("Training Classifier (This may take a while)..."):
                        train_audio_model()
                    st.success("分类模型已更新")
            
            with c2:
                st.subheader("2. 距离融合模型")
                st.caption("架构: FusionCRNN | 特征: Log-Mel + Stats")
                if st.button("🚀 启动距离训练"):
                    with st.spinner("Training Fusion Model..."):
                        train_distance_model()
                    st.success("距离模型已更新")

else:
    # 欢迎页
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🔫 PUBG 智能战术终端 Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Powered by PyTorch & Streamlit</p>", unsafe_allow_html=True)
    st.info("👈 请在左侧侧边栏登录或注册以开始使用。")
