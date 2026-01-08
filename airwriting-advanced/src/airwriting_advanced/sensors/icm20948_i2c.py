from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .imu_base import IMUSample, IMUDevice
from .linux_i2c import LinuxI2C


@dataclass
class ICM20948Settings:
    sample_rate_hz: float = 200.0
    accel_fs_g: int = 4      # 2,4,8,16
    gyro_fs_dps: int = 500   # 250,500,1000,2000
    dlpf_cfg: int = 3        # 0~7 (datasheet table)
    enable_mag: bool = True
    mag_mode: int = 0x08     # AK09916 continuous mode4 (100Hz)


class ICM20948I2C(IMUDevice):
    """
    ICM-20948 (acc+gyro) + AK09916 (mag) via BYPASS mode.
    - Main device i2c address: 0x68 or 0x69
    - Magnetometer i2c address (bypass): 0x0C
    """
    name: str

    # bank select
    REG_BANK_SEL = 0x7F

    # bank 0
    REG_WHO_AM_I = 0x00
    REG_USER_CTRL = 0x03
    REG_PWR_MGMT_1 = 0x06
    REG_PWR_MGMT_2 = 0x07
    REG_INT_PIN_CFG = 0x0F
    REG_ACCEL_XOUT_H = 0x2D  # 0x2D..0x3A covers accel/gyro/temp

    WHO_AM_I_EXPECTED = 0xEA

    # bank 2 (common locations)
    REG_GYRO_SMPLRT_DIV = 0x00
    REG_GYRO_CONFIG_1 = 0x01
    REG_ACCEL_SMPLRT_DIV_1 = 0x10
    REG_ACCEL_SMPLRT_DIV_2 = 0x11
    REG_ACCEL_CONFIG = 0x14

    # AK09916
    MAG_ADDR = 0x0C
    MAG_REG_WIA2 = 0x01
    MAG_WIA2_EXPECTED = 0x09
    MAG_REG_ST1 = 0x10
    MAG_REG_HXL = 0x11
    MAG_REG_CNTL2 = 0x31
    MAG_REG_CNTL3 = 0x32

    def __init__(self, name: str, bus: int, addr: int, settings: ICM20948Settings):
        self.name = name
        self.bus = int(bus)
        self.addr = int(addr)
        self.settings = settings

        self.i2c = LinuxI2C(self.bus, self.addr)
        self.i2c.open()

        # scales
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

        self.mag_i2c: Optional[LinuxI2C] = None
        self._init_device()

    def _select_bank(self, bank: int) -> None:
        self.i2c.write_u8(self.REG_BANK_SEL, (bank & 0x03) << 4)

    def _init_device(self) -> None:
        # check whoami
        self._select_bank(0)
        who = self.i2c.read_u8(self.REG_WHO_AM_I)
        if who != self.WHO_AM_I_EXPECTED:
            # still proceed; some breakouts return 0xEA but allow mismatch for clones
            pass

        # wake: auto clock
        self.i2c.write_u8(self.REG_PWR_MGMT_1, 0x01)
        time.sleep(0.02)
        self.i2c.write_u8(self.REG_PWR_MGMT_2, 0x00)

        # configure accel/gyro (bank2)
        self._select_bank(2)

        # sample rate dividers (base rate 1.125kHz) - keep simple
        div = max(0, int(round(1125.0 / float(self.settings.sample_rate_hz) - 1.0)))
        div = min(div, 255)
        self.i2c.write_u8(self.REG_GYRO_SMPLRT_DIV, div)

        adiv = max(0, int(round(1125.0 / float(self.settings.sample_rate_hz) - 1.0)))
        adiv = min(adiv, 4095)
        self.i2c.write_u8(self.REG_ACCEL_SMPLRT_DIV_1, (adiv >> 8) & 0x0F)
        self.i2c.write_u8(self.REG_ACCEL_SMPLRT_DIV_2, adiv & 0xFF)

        # gyro config: DLPFCFG[5:3], FS_SEL[2:1], FCHOICE[0]=1(enable DLPF)
        fs_sel = {250: 0, 500: 1, 1000: 2, 2000: 3}[int(self.settings.gyro_fs_dps)]
        gyro_cfg = ((self.settings.dlpf_cfg & 0x07) << 3) | ((fs_sel & 0x03) << 1) | 0x01
        self.i2c.write_u8(self.REG_GYRO_CONFIG_1, gyro_cfg)

        # accel config: DLPFCFG[5:3], FS_SEL[2:1], FCHOICE[0]=1
        afs_sel = {2: 0, 4: 1, 8: 2, 16: 3}[int(self.settings.accel_fs_g)]
        accel_cfg = ((self.settings.dlpf_cfg & 0x07) << 3) | ((afs_sel & 0x03) << 1) | 0x01
        self.i2c.write_u8(self.REG_ACCEL_CONFIG, accel_cfg)

        # back to bank0
        self._select_bank(0)

        # enable bypass for magnetometer (AK09916 on 0x0C)
        if self.settings.enable_mag:
            # disable I2C master
            self.i2c.write_u8(self.REG_USER_CTRL, 0x00)
            time.sleep(0.01)
            # enable bypass (BYPASS_EN bit)
            self.i2c.write_u8(self.REG_INT_PIN_CFG, 0x02)
            time.sleep(0.01)

            self.mag_i2c = LinuxI2C(self.bus, self.MAG_ADDR)
            self.mag_i2c.open()

            # check mag whoami
            wia2 = self.mag_i2c.read_u8(self.MAG_REG_WIA2)
            if wia2 != self.MAG_WIA2_EXPECTED:
                # still proceed, but mark as disabled
                self.mag_i2c.close()
                self.mag_i2c = None
                return

            # soft reset
            self.mag_i2c.write_u8(self.MAG_REG_CNTL3, 0x01)
            time.sleep(0.01)

            # set continuous mode (100Hz default)
            self.mag_i2c.write_u8(self.MAG_REG_CNTL2, self.settings.mag_mode & 0x1F)
            time.sleep(0.01)

    @staticmethod
    def _s16be(b: bytes, off: int) -> int:
        v = (b[off] << 8) | b[off + 1]
        return v - 65536 if v & 0x8000 else v

    @staticmethod
    def _s16le(b: bytes, off: int) -> int:
        v = b[off] | (b[off + 1] << 8)
        return v - 65536 if v & 0x8000 else v

    def _read_mag_uT(self) -> Optional[np.ndarray]:
        if self.mag_i2c is None:
            return None

        # check data ready
        st1 = self.mag_i2c.read_u8(self.MAG_REG_ST1)
        if (st1 & 0x01) == 0:
            return None

        # read HXL..HZH + TMPS + ST2 (8 bytes)
        b = self.mag_i2c.read_n(self.MAG_REG_HXL, 8)
        hx = self._s16le(b, 0)
        hy = self._s16le(b, 2)
        hz = self._s16le(b, 4)
        st2 = b[7]
        if (st2 & 0x08) != 0:
            # overflow
            return None

        # 0.15 uT/LSB (typ.)
        return np.array([hx, hy, hz], dtype=float) * 0.15

    def read(self) -> IMUSample:
        t = time.time()

        self._select_bank(0)
        b = self.i2c.read_n(self.REG_ACCEL_XOUT_H, 14)

        ax = self._s16be(b, 0) / self._acc_lsb_per_g * 9.80665
        ay = self._s16be(b, 2) / self._acc_lsb_per_g * 9.80665
        az = self._s16be(b, 4) / self._acc_lsb_per_g * 9.80665

        gx = self._s16be(b, 6) / self._gyr_lsb_per_dps * (np.pi / 180.0)
        gy = self._s16be(b, 8) / self._gyr_lsb_per_dps * (np.pi / 180.0)
        gz = self._s16be(b, 10) / self._gyr_lsb_per_dps * (np.pi / 180.0)

        temp_raw = self._s16be(b, 12)
        # rough conversion (datasheet formula can differ; keep for debug only)
        temp_C = (temp_raw / 333.87) + 21.0

        mag_uT = self._read_mag_uT()

        return IMUSample(
            t=t,
            acc_m_s2=np.array([ax, ay, az], dtype=float),
            gyr_rad_s=np.array([gx, gy, gz], dtype=float),
            mag_uT=mag_uT,
            temp_C=float(temp_C),
        )

    def close(self) -> None:
        try:
            if self.mag_i2c is not None:
                self.mag_i2c.close()
        finally:
            self.i2c.close()
