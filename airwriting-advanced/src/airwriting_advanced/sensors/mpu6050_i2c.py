from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .imu_base import IMUSample, IMUDevice
from .linux_i2c import LinuxI2C


@dataclass
class MPU6050Settings:
    sample_rate_hz: float = 200.0
    accel_fs_g: int = 4      # 2,4,8,16
    gyro_fs_dps: int = 500   # 250,500,1000,2000
    dlpf_cfg: int = 3        # 0~6 (0:260Hz, 3:44Hz)


class MPU6050I2C(IMUDevice):
    """
    MPU-6050 (acc+gyro) driver.
    - i2c address: 0x68 (AD0 low) or 0x69 (AD0 high)
    - outputs: accel [m/s^2], gyro [rad/s]
    """
    name: str

    # registers
    REG_PWR_MGMT_1 = 0x6B
    REG_SMPLRT_DIV = 0x19
    REG_CONFIG = 0x1A
    REG_GYRO_CONFIG = 0x1B
    REG_ACCEL_CONFIG = 0x1C
    REG_INT_ENABLE = 0x38
    REG_ACCEL_XOUT_H = 0x3B

    def __init__(self, name: str, bus: int, addr: int, settings: MPU6050Settings):
        self.name = name
        self.bus = int(bus)
        self.addr = int(addr)
        self.settings = settings
        self.i2c = LinuxI2C(self.bus, self.addr)
        self.i2c.open()
        self._init_device()

        self._acc_lsb_per_g = {
            2: 16384.0,
            4: 8192.0,
            8: 4096.0,
            16: 2048.0,
        }[int(settings.accel_fs_g)]
        self._gyr_lsb_per_dps = {
            250: 131.0,
            500: 65.5,
            1000: 32.8,
            2000: 16.4,
        }[int(settings.gyro_fs_dps)]

    def _init_device(self) -> None:
        # wake up
        self.i2c.write_u8(self.REG_PWR_MGMT_1, 0x00)
        time.sleep(0.02)

        # set DLPF
        self.i2c.write_u8(self.REG_CONFIG, self.settings.dlpf_cfg & 0x07)

        # sample rate div (assume 1kHz internal when DLPF enabled)
        div = max(0, int(round(1000.0 / float(self.settings.sample_rate_hz) - 1.0)))
        if div > 255:
            div = 255
        self.i2c.write_u8(self.REG_SMPLRT_DIV, div)

        # gyro full scale
        fs_sel = {250: 0, 500: 1, 1000: 2, 2000: 3}[int(self.settings.gyro_fs_dps)]
        self.i2c.write_u8(self.REG_GYRO_CONFIG, (fs_sel & 0x03) << 3)

        # accel full scale
        afs_sel = {2: 0, 4: 1, 8: 2, 16: 3}[int(self.settings.accel_fs_g)]
        self.i2c.write_u8(self.REG_ACCEL_CONFIG, (afs_sel & 0x03) << 3)

        # data ready interrupt (optional)
        self.i2c.write_u8(self.REG_INT_ENABLE, 0x01)

    @staticmethod
    def _s16be(b: bytes, off: int) -> int:
        v = (b[off] << 8) | b[off + 1]
        return v - 65536 if v & 0x8000 else v

    def read(self) -> IMUSample:
        t = time.time()
        b = self.i2c.read_n(self.REG_ACCEL_XOUT_H, 14)

        ax = self._s16be(b, 0) / self._acc_lsb_per_g * 9.80665
        ay = self._s16be(b, 2) / self._acc_lsb_per_g * 9.80665
        az = self._s16be(b, 4) / self._acc_lsb_per_g * 9.80665

        temp_raw = self._s16be(b, 6)
        temp_C = temp_raw / 340.0 + 36.53

        gx = self._s16be(b, 8) / self._gyr_lsb_per_dps * (np.pi / 180.0)
        gy = self._s16be(b, 10) / self._gyr_lsb_per_dps * (np.pi / 180.0)
        gz = self._s16be(b, 12) / self._gyr_lsb_per_dps * (np.pi / 180.0)

        return IMUSample(
            t=t,
            acc_m_s2=np.array([ax, ay, az], dtype=float),
            gyr_rad_s=np.array([gx, gy, gz], dtype=float),
            mag_uT=None,
            temp_C=float(temp_C),
        )

    def close(self) -> None:
        self.i2c.close()
