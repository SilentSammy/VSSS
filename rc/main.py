"""
BLE Remote Control for MecanumCar
Controls X, Y, W axes via BLE characteristics
"""

import asyncio
from mecanum_client import control_loop

if __name__ == "__main__":
    try:
        asyncio.run(control_loop())
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Error: {e}")
