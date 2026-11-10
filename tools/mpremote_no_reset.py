"""Run mpremote without toggling native ESP32 USB reset/boot lines on open."""

import sys

import serial
from mpremote import main


_serial_for_url = serial.serial_for_url


def _open_idle(*args, **kwargs):
    port = _serial_for_url(*args, **kwargs)
    # Native ESP32-C3 USB Serial/JTAG maps these to BOOT and RESET.  pyserial's
    # normal defaults assert both when opening, which restarts a running board.
    port.dtr = False
    port.rts = False
    return port


serial.serial_for_url = _open_idle

if __name__ == "__main__":
    main.main()
