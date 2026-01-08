from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from smbus2 import SMBus

from .imu_base import IMUDevice, IMUSample


# --- MPU-9250 register map (MPU core) ---
MPU_WHO_AM_I = 0x75
MPU_PWR_MGMT_1 = 0x6B
MPU_PWR_MGMT_2 = 0x6C
MPU_SMPLRT_DIV = 0x19
MPU_CONFIG = 0x1A
MPU_GYRO_CONFIG = 0x1B
MPU_ACCEL_CONFIG = 0x1C
MPU_ACCEL_CONFIG2 = 0x1D
MPU_INT_PIN_CFG = 0x37

MPU_ACCEL_XOUT_H = 0x3B
MPU_TEMP_OUT_H = 0x41
MPU_GYRO_XOUT_H = 0x43

# --- AK8963 magnetometer ---
AK8963_ADDR = 0x0C
AK8963_WHO_AM_I = 0x00
AK8963_ST1 = 0x02
AK8963_HXL = 0x03
AK8963_CNTL1 = 0x0A
AK8963_ASAX = 0x10

# Scale factors (based on selected full-scale ranges)
# Defaults below: accel ±2g (16384 LSB/g), gyro ±2000 dps (16.4 LSB/(dps))
ACC_LSB_PER_G = 16384.0
GYR_LSB_PER_DPS = 16.4

G = 9.80665
DEG2RAD = np.pi / 180.0

# AK8963: 16-bit output scale ~0.15 uT/LSB (4912 uT full scale)
MAG_UT_PER_LSB_16BIT = 4912.0 / 32760.0  # ~0.1499


@dataclass
class MPU9250Settings:
    sample_rate_hz: float = 100.0
    gyro_range_dps: int = 2000  # 250/500/1000/2000
    accel_range_g: int = 2      # 2/4/8/16
    enable_mag: bool = False


class MPU9250I2C(IMUDevice):
    def __init__(self, name: str, bus: int, addr: int, settings: Optional[MPU9250Settings] = None):
        self.name = name
        self.bus_id = int(bus)
        self.addr = int(addr)
        self.bus = SMBus(self.bus_id)
        self.settings = settings or MPU9250Settings()
        self._acc_bias = np.zeros(3, dtype=float)
        self._gyr_bias = np.zeros(3, dtype=float)
        self._mag_adjust = np.ones(3, dtype=float)

        self._init_mpu()
        if self.settings.enable_mag:
            self._init_mag()

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass

    def _read_u8(self, reg: int) -> int:
        return int(self.bus.read_byte_data(self.addr, reg))

    def _write_u8(self, reg: int, val: int) -> None:
        self.bus.write_byte_data(self.addr, reg, val & 0xFF)

    def _read_block(self, reg: int, length: int) -> bytes:
        data = self.bus.read_i2c_block_data(self.addr, reg, length)
        return bytes(data)

    def _init_mpu(self) -> None:
        # Wake up
        self._write_u8(MPU_PWR_MGMT_1, 0x00)
        time.sleep(0.05)

        # Basic config
        # DLPF config
        self._write_u8(MPU_CONFIG, 0x03)  # ~44 Hz gyro DLPF
        self._write_u8(MPU_ACCEL_CONFIG2, 0x03)  # ~44 Hz accel DLPF

        # Sample rate
        # sample_rate = gyro_output_rate / (1 + div), gyro_output_rate=1kHz when DLPF enabled
        target = max(1.0, float(self.settings.sample_rate_hz))
        div = int(round(1000.0 / target - 1.0))
        div = max(0, min(255, div))
        self._write_u8(MPU_SMPLRT_DIV, div)

        # Gyro full scale
        gyr_fs = self.settings.gyro_range_dps
        gyr_bits = {250: 0, 500: 1, 1000: 2, 2000: 3}.get(gyr_fs, 3)
        self._write_u8(MPU_GYRO_CONFIG, gyr_bits << 3)

        # Accel full scale
        acc_fs = self.settings.accel_range_g
        acc_bits = {2: 0, 4: 1, 8: 2, 16: 3}.get(acc_fs, 0)
        self._write_u8(MPU_ACCEL_CONFIG, acc_bits << 3)

        # Enable bypass to access magnetometer if needed
        if self.settings.enable_mag:
            self._write_u8(MPU_INT_PIN_CFG, 0x02)  # BYPASS_EN

    def _init_mag(self) -> None:
        # Read factory sensitivity adjustment values
        who = self.bus.read_byte_data(AK8963_ADDR, AK8963_WHO_AM_I)
        # Set to power-down
        self.bus.write_byte_data(AK8963_ADDR, AK8963_CNTL1, 0x00)
        time.sleep(0.01)
        # Enter fuse ROM access mode
        self.bus.write_byte_data(AK8963_ADDR, AK8963_CNTL1, 0x0F)
        time.sleep(0.01)
        asax = self.bus.read_i2c_block_data(AK8963_ADDR, AK8963_ASAX, 3)
        # sensitivity adjustment per datasheet: adj = (ASA-128)/256 + 1
        self._mag_adjust = np.array([(a - 128) / 256.0 + 1.0 for a in asax], dtype=float)
        # Power-down
        self.bus.write_byte_data(AK8963_ADDR, AK8963_CNTL1, 0x00)
        time.sleep(0.01)
        # Continuous measurement mode 2 (100Hz), 16-bit output
        self.bus.write_byte_data(AK8963_ADDR, AK8963_CNTL1, 0x16)
        time.sleep(0.01)

    def calibrate_bias(self, seconds: float = 2.0) -> None:
        """Average accel/gyro at rest; accel bias includes gravity in current orientation."""
        t0 = time.time()
        accs = []
        gyrs = []
        while time.time() - t0 < seconds:
            s = self.read()
            accs.append(s.acc_m_s2)
            gyrs.append(s.gyr_rad_s)
            time.sleep(0.01)
        if accs:
            self._acc_bias = np.mean(np.stack(accs, axis=0), axis=0)
        if gyrs:
            self._gyr_bias = np.mean(np.stack(gyrs, axis=0), axis=0)

    def _read_acc_gyr_temp_raw(self) -> tuple[np.ndarray, np.ndarray, int]:
        data = self._read_block(MPU_ACCEL_XOUT_H, 14)
        ax, ay, az, temp, gx, gy, gz = struct.unpack(">hhhhhhh", data)
        acc_raw = np.array([ax, ay, az], dtype=float)
        gyr_raw = np.array([gx, gy, gz], dtype=float)
        return acc_raw, gyr_raw, int(temp)

    def _read_mag_raw(self) -> Optional[np.ndarray]:
        # check data ready
        st1 = self.bus.read_byte_data(AK8963_ADDR, AK8963_ST1)
        if (st1 & 0x01) == 0:
            return None
        data = self.bus.read_i2c_block_data(AK8963_ADDR, AK8963_HXL, 7)
        hx = data[0] | (data[1] << 8)
        hy = data[2] | (data[3] << 8)
        hz = data[4] | (data[5] << 8)
        st2 = data[6]
        # overflow
        if (st2 & 0x08) != 0:
            return None
        # convert to signed 16-bit
        def s16(x):
            return x - 65536 if x >= 32768 else x
        return np.array([s16(hx), s16(hy), s16(hz)], dtype=float)

    def read(self) -> IMUSample:
        t = time.time()
        acc_raw, gyr_raw, temp_raw = self._read_acc_gyr_temp_raw()

        # Convert scales
        # Adjust LSB per g based on selected range
        acc_fs = self.settings.accel_range_g
        acc_lsb_per_g = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}.get(acc_fs, 16384.0)
        gyr_fs = self.settings.gyro_range_dps
        gyr_lsb_per_dps = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}.get(gyr_fs, 16.4)

        acc = (acc_raw / acc_lsb_per_g) * G
        gyr = (gyr_raw / gyr_lsb_per_dps) * DEG2RAD

        # Temperature: datasheet formula approx: Temp(C) = (temp_raw / 333.87) + 21
        temp_C = (temp_raw / 333.87) + 21.0

        mag = None
        if self.settings.enable_mag:
            mr = self._read_mag_raw()
            if mr is not None:
                mag = (mr * self._mag_adjust) * MAG_UT_PER_LSB_16BIT

        # Apply biases
        acc = acc - self._acc_bias
        gyr = gyr - self._gyr_bias

        return IMUSample(t=t, acc_m_s2=acc, gyr_rad_s=gyr, mag_uT=mag, temp_C=temp_C)
