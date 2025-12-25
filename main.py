import sys
import time
import hashlib
from player import Player
from weapon import Weapon
from storage import load_players, save_players, load_global_weapon_library, load_admins

# === 导入 AI 模块 (保持你的模块不动) ===
from model_trainer import train_audio_model
from distance_trainer import train_distance_model
from sound_inference import predict_user_audio

# --- 辅助输入函数 ---
try:
    import stdiomask
    def secure_input(prompt): return stdiomask.getpass(prompt=prompt, mask='*')
except:
    def secure_input(prompt): return input(prompt)

def safe_password_input(prompt):
    try: return secure_input(prompt)
    except: return input(prompt)

# --- 业务逻辑函数 ---

def register_player(players, global_weapons):
    print("\n=== 新兵注册 ===")
    while True:
        sid = input("请输入学号: ").strip()
        if not sid: continue
        if sid in players:
            print("❌ 该学号已存在。")
            return
        break

    while True:
        p1 = safe_password_input("请设置密码: ")
        if not p1: continue
        p2 = safe_password_input("请确认密码: ")
        if p1 == p2: break
        print("❌ 密码不一致。")

    new_user = Player(sid)
    new_user.set_password(p1)

    if global_weapons:
        print(f"\n📦 正在领取 {len(global_weapons)} 件初始武器...")
        for w_obj in global_weapons.values():
            new_user.add_weapon(w_obj)
        print("✅ 武器配发完毕！")
    else:
        print("\n⚠️ 警告: 武器库为空。")

    players[sid] = new_user
    save_players(players)
    print(f"✅ 注册成功！")
    time.sleep(1)

def add_custom_weapon(user, players):
    print("\n🛠️ --- 制造新武器 ---")
    try:
        name = input("名称: ").strip()
        dmg = int(input("伤害: "))
        mode = input("模式: ").strip()
        cap = int(input("容量: "))
        res = int(input("备弹数: "))
        cur = int(input("枪膛子弹: "))

        new_w = Weapon(name, dmg, mode, cap, res, cur)
        user.add_weapon(new_w)
        save_players(players)
        print(f"✅ {name} 已加入背包！")
    except ValueError:
        print("❌ 输入错误：数字项必须填整数！")

def modify_weapon_attributes(user, players):
    if not user.weapons:
        print("背包为空。")
        return

    idx_str = input("请输入要改装的武器序号 (No.): ")
    if not idx_str.isdigit():
        print("❌ 请输入数字。")
        return

    idx = int(idx_str) - 1
    if not (0 <= idx < len(user.weapons)):
        print("❌ 序号无效。")
        return

    target_w = user.weapons[idx]
    print(f"\n🔧 正在改装: {target_w.name}")
    print("1. 修改名称")
    print("2. 调整枪膛内子弹")
    print("3. 调整备用弹夹数")
    print("0. 取消")
    
    opt = input("选择: ").strip()

    if opt == "1":
        new_name = input("输入新名称: ").strip()
        if new_name:
            target_w.name = new_name
            save_players(players)
            print("✅ 修改成功")
    elif opt == "2":
        try:
            val = int(input(f"新数值 (Max {target_w.magazine_capacity}): "))
            if 0 <= val <= target_w.magazine_capacity:
                target_w.current_ammo = val
                save_players(players)
                print("✅ 修改成功")
            else: print("❌ 超限")
        except: print("❌ 输入无效")
    elif opt == "3":
        try:
            val = int(input("新数值: "))
            if val >= 0:
                target_w.reserve_mags = val
                save_players(players)
                print("✅ 修改成功")
        except: print("❌ 输入无效")

def sort_logic(user, players):
    print("\n📊 --- 武器库排序 ---")
    c = input("1. 伤害从高到低\n2. 伤害从低到高\n选择: ")
    if c == "1":
        user.sort_weapons_by_damage(reverse=True)
        print("✅ 已降序排列")
        save_players(players)
    elif c == "2":
        user.sort_weapons_by_damage(reverse=False)
        print("✅ 已升序排列")
        save_players(players)

def delete_weapon(user, players):
    if not user.weapons:
        print("背包为空。")
        return
    idx_str = input("输入序号丢弃: ")
    if idx_str.isdigit():
        idx = int(idx_str) - 1
        if user.remove_weapon_by_index(idx):
            save_players(players)
            print("🗑️ 已丢弃")
        else: print("❌ 序号无效")

def delete_account(user, players):
    print("\n⚠️ 警告：正在注销账号")
    if input("确认 (输入yes): ").strip().lower() == "yes":
        pwd = safe_password_input("验证密码: ")
        if user.check_password(pwd):
            del players[user.student_id]
            save_players(players)
            print("💀 账号已注销")
            return True
        else:
            print("❌ 密码错误")
    return False

# --- 新增功能模块 ---

def search_weapon(user):
    """武器搜索功能"""
    print("\n🔍 --- 武器库搜索 ---")
    query = input("请输入搜索关键词 (如 AK): ").strip().lower()
    if not query: return
    
    found = [w for w in user.weapons if query in w.name.lower()]
    
    if found:
        print(f"\n✅ 找到 {len(found)} 把相关武器:")
        print(f"{'No.':<4} {'名称':<10} {'伤害':<6} {'模式':<10} {'弹药'}")
        print("-" * 50)
        # 这里为了显示方便，重新在原列表中找索引有点麻烦，直接显示内容
        for w in found:
            print(f"{'--':<4} {w.name:<10} {w.damage:<6} {w.firing_mode:<10} {w.current_ammo}/{w.reserve_mags*w.magazine_capacity}")
    else:
        print("📭 未找到匹配的武器。")

def admin_panel():
    """管理员专属面板"""
    print("\n🔒 --- 管理员后台 ---")
    admins = load_admins()
    if not admins: 
        print("❌ 错误：admins.json 缺失")
        return
        
    aid = input("Admin ID: ")
    if aid not in admins:
        print("❌ 账号错误")
        return
        
    pwd = safe_password_input("Password: ")
    if hashlib.sha256(pwd.encode()).hexdigest() == admins[aid]:
        print("✅ 验证通过")
        while True:
            print("\n--- AI 训练中心 ---")
            print("1. 训练枪声分类模型 (1D-CNN + Class Weights)")
            print("2. 训练距离感知模型 (Fusion-CRNN + Classification)")
            print("0. 返回主菜单")
            c = input("指令: ")
            if c == "1": train_audio_model()
            elif c == "2": train_distance_model()
            elif c == "0": break
    else:
        print("❌ 密码错误")

def login_and_manage(players):
    print("\n=== 用户登录 ===")
    sid = input("学号: ").strip()
    if sid not in players:
        print("❌ 用户不存在")
        return
    
    user = players[sid]
    if user.check_password(safe_password_input("密码: ")):
        print(f"✅ 欢迎回来 {sid}")
        while True:
            print(f"\n--- 指挥官菜单 (背包: {len(user.weapons)}) ---")
            print("1. 制造武器   2. 改装武器")
            print("3. 丢弃武器   4. 排序背包")
            print("5. 🔍 搜索武器")
            print("6. 🎤 识别枪声 (AI双模)")
            print("7. 🔒 管理员后台 (训练)")
            print("99.注销       0.退出")
            
            c = input("指令: ")
            if c == "1": add_custom_weapon(user, players)
            elif c == "2": modify_weapon_attributes(user, players)
            elif c == "3": delete_weapon(user, players)
            elif c == "4": sort_logic(user, players)
            elif c == "5": search_weapon(user)
            elif c == "6": predict_user_audio()
            elif c == "7": admin_panel()
            elif c == "99": 
                if delete_account(user, players): break
            elif c == "0": break
            else: print("无效指令")
    else:
        print("❌ 密码错误")

def main():
    print("🚀 PUBG 武器系统 v10.0 (Ultimate CLI) 启动...")
    # 初始化加载
    global_weapons = load_global_weapon_library()
    players = load_players()
    
    while True:
        print("\n1. 注册新兵")
        print("2. 登录系统")
        print("0. 退出程序")
        choice = input("选择: ").strip()
        
        if choice == "1":
            register_player(players, global_weapons)
        elif choice == "2":
            login_and_manage(players)
        elif choice == "0":
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序终止。")
    except Exception as e:
        print(f"Error: {e}")
        input("Enter退出")
