# weapon.py
class Weapon:
    def __init__(self, name, damage, firing_mode, magazine_capacity, reserve_mags, current_ammo):
        self.name = name
        self.damage = int(damage)
        self.firing_mode = firing_mode
        self.magazine_capacity = int(magazine_capacity)
        
        # 直接使用 CSV 里的数据
        self.reserve_mags = int(reserve_mags)
        self.current_ammo = int(current_ammo)

    @property
    def total_ammo(self):
        """计算总战斗力：枪内 + 备弹 * 弹夹容量"""
        return self.current_ammo + (self.reserve_mags * self.magazine_capacity)

    def to_dict(self):
        """序列化保存到 json"""
        return {
            "name": self.name,
            "damage": self.damage,
            "firing_mode": self.firing_mode,
            "magazine_capacity": self.magazine_capacity,
            "reserve_mags": self.reserve_mags,
            "current_ammo": self.current_ammo
        }

    @classmethod
    def from_dict(cls, data):
        """从 json 读取"""
        return cls(
            name=data["name"],
            damage=data["damage"],
            firing_mode=data["firing_mode"],
            magazine_capacity=data["magazine_capacity"],
            reserve_mags=data.get("reserve_mags", 0),
            current_ammo=data.get("current_ammo", 0)
        )

    def __str__(self):
        return f"[{self.name}] 攻:{self.damage} 弹:{self.current_ammo} (备:{self.reserve_mags})"
