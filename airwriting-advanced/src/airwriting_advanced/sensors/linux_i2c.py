from __future__ import annotations

import os
import fcntl
from dataclasses import dataclass
from typing import Optional

# Linux i2c-dev ioctl
I2C_SLAVE = 0x0703


@dataclass
class LinuxI2C:
    """
    Minimal i2c-dev helper (no external deps).
    - Uses /dev/i2c-X
    - Supports simple register read/write (8-bit register address).
    """
    bus: int
    addr: int
    fd: Optional[int] = None

    def open(self) -> None:
        if self.fd is not None:
            return
        path = f"/dev/i2c-{self.bus}"
        self.fd = os.open(path, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, self.addr)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write(self, data: bytes) -> None:
        if self.fd is None:
            self.open()
        assert self.fd is not None
        os.write(self.fd, data)

    def read(self, n: int) -> bytes:
        if self.fd is None:
            self.open()
        assert self.fd is not None
        return os.read(self.fd, n)

    def write_u8(self, reg: int, val: int) -> None:
        self.write(bytes([reg & 0xFF, val & 0xFF]))

    def read_u8(self, reg: int) -> int:
        self.write(bytes([reg & 0xFF]))
        return self.read(1)[0]

    def read_n(self, reg: int, n: int) -> bytes:
        self.write(bytes([reg & 0xFF]))
        return self.read(n)
