# storage.py
import csv
import json
from pathlib import Path
from player import Player
from weapon import Weapon

BASE_DIR = Path(__file__).resolve().parent
PLAYERS_FILE = BASE_DIR / "players.json"
ARMS_FILE = BASE_DIR / "Arms.csv"
ADMINS_FILE = BASE_DIR / "admins.json"  # 👈 新增：管理员文件路径

def load_global_weapon_library() -> dict:
    if not ARMS_FILE.exists():
        print(f"❌ 错误: 未找到 Arms.csv ({ARMS_FILE})")
        return {}

    weapons = {}
    encoding_list = ['utf-8-sig', 'gbk', 'utf-8']
    
    csv_file = None
    for enc in encoding_list:
        try:
            csv_file = open(ARMS_FILE, "r", encoding=enc)
            csv_file.read(10)
            csv_file.seek(0)
            break
        except:
            if csv_file: csv_file.close()
            continue

    if not csv_file: return {}

    try:
        reader = csv.reader(csv_file)
        next(reader, None)
        
        for row in reader:
            if not row or len(row) < 7: continue
            try:
                w_name = row[1].strip()
                w = Weapon(
                    name=w_name,
                    damage=row[2].strip(),
                    firing_mode=row[3].strip(),
                    magazine_capacity=row[4].strip(),
                    reserve_mags=row[5].strip(),
                    current_ammo=row[6].strip()
                )
                weapons[w_name] = w
            except ValueError: continue
        return weapons
    except:
        return {}
    finally:
        csv_file.close()

def load_players() -> dict:
    if not PLAYERS_FILE.exists(): return {}
    try:
        with open(PLAYERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {pid: Player.from_dict(pdata) for pid, pdata in data.items()}
    except: return {}

def save_players(players: dict) -> bool:
    try:
        data = {pid: p.to_dict() for pid, p in players.items()}
        with open(PLAYERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except: return False

# 👇 新增：读取管理员列表
def load_admins() -> dict:
    if not ADMINS_FILE.exists():
        print("⚠️ 警告：未找到 admins.json，无法进行管理员验证。")
        return {}
    try:
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 管理员文件读取失败: {e}")
        return {}
