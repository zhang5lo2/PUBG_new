# player.py
import hashlib
import copy
from weapon import Weapon

class Player:
    def __init__(self, student_id):
        self.student_id = student_id
        self.password_hash = ""
        self.weapons = []  # 存储 Weapon 对象列表

    def set_password(self, plain_password):
        """密码加密存储"""
        self.password_hash = hashlib.sha256(plain_password.encode()).hexdigest()

    def check_password(self, plain_password):
        """验证密码"""
        input_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        return input_hash == self.password_hash

    def add_weapon(self, weapon_template: Weapon):
        """
        核心逻辑：深拷贝。
        确保玩家拿到的是一把新枪，而不是引用了公共武器库的对象。
        """
        new_w = copy.deepcopy(weapon_template)
        self.weapons.append(new_w)

    def remove_weapon_by_index(self, index):
        """
        根据列表索引删除武器
        index: 从 0 开始的索引
        """
        if 0 <= index < len(self.weapons):
            return self.weapons.pop(index)
        return None

    def sort_weapons_by_damage(self, reverse=True):
        """按伤害排序 (默认从高到低)"""
        self.weapons.sort(key=lambda w: w.damage, reverse=reverse)

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "password_hash": self.password_hash,
            "weapons": [w.to_dict() for w in self.weapons]
        }

    @classmethod
    def from_dict(cls, data):
        p = cls(data["student_id"])
        p.password_hash = data["password_hash"]
        p.weapons = [Weapon.from_dict(w) for w in data["weapons"]]
        return p
