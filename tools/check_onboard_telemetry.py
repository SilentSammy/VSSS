"""Read-only ESP32 diagnostic; run with mpremote, never initializes motors."""

import network
import time

from pose_receiver import PoseReceiver
from robot_config import RobotConfig


settings = RobotConfig().data
wlan = network.WLAN(network.STA_IF)
if not wlan.isconnected():
    wlan.active(True)
    wlan.connect(settings["wifi_ssid"], settings["wifi_password"])
    deadline = time.ticks_add(time.ticks_ms(), 10000)
    while not wlan.isconnected() and time.ticks_diff(deadline, time.ticks_ms()) > 0:
        time.sleep_ms(50)

print("wifi", wlan.isconnected(), wlan.ifconfig() if wlan.isconnected() else None)
receiver = PoseReceiver(settings["car_id"], subscribe=False)
deadline = time.ticks_add(time.ticks_ms(), 5000)
while time.ticks_diff(deadline, time.ticks_ms()) > 0:
    receiver.poll()
    time.sleep_ms(10)
print("packets", receiver.packets)
print("fresh", receiver.fresh)
print("pose", receiver.pose)
receiver.close()
