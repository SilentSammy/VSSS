"""Foreground MicroPython waypoint diagnostic, intended for ``mpremote run``."""

import math
import network
import time

from axis_trim import AxisTrim
from mecanum import MecanumCar
from onboard_controller import OnboardController
from pose_receiver import PoseReceiver
from robot_config import RobotConfig


TARGET = (0.0, 0.0, 90.0)
TIMEOUT_MS = 25000


settings = RobotConfig().data
car = MecanumCar()
car.stop()
receiver = None

try:
    wlan = network.WLAN(network.STA_IF)
    if not wlan.isconnected():
        wlan.active(True)
        wlan.connect(settings["wifi_ssid"], settings["wifi_password"])
        deadline = time.ticks_add(time.ticks_ms(), 10000)
        while not wlan.isconnected() and time.ticks_diff(deadline, time.ticks_ms()) > 0:
            time.sleep_ms(50)
    if not wlan.isconnected():
        raise RuntimeError("WiFi connection failed")

    receiver = PoseReceiver(settings["car_id"], subscribe=False)
    wait_deadline = time.ticks_add(time.ticks_ms(), 5000)
    while not receiver.fresh and time.ticks_diff(wait_deadline, time.ticks_ms()) > 0:
        receiver.poll()
        time.sleep_ms(10)
    if not receiver.fresh:
        raise RuntimeError("No fresh marker telemetry")

    trim = AxisTrim(settings["trim_matrix"])
    controller = OnboardController(**settings["controller"])
    controller.set_target(TARGET[0], TARGET[1], math.radians(TARGET[2]))
    deadline = time.ticks_add(time.ticks_ms(), TIMEOUT_MS)
    next_report = 0
    print("start", receiver.pose, "target", TARGET)

    while time.ticks_diff(deadline, time.ticks_ms()) > 0:
        receiver.poll()
        now = time.ticks_ms()
        if not receiver.fresh:
            raise RuntimeError("Marker telemetry became stale")
        command = controller.update(receiver.pose, now)
        car.set_velocity(*trim.apply(command))
        if time.ticks_diff(now, next_report) >= 0:
            print("pose", receiver.pose, "command", command)
            next_report = time.ticks_add(now, 250)
        if controller.reached(now):
            print("reached", receiver.pose)
            break
        time.sleep_ms(10)
    else:
        raise RuntimeError("Waypoint timed out")
finally:
    car.stop()
    if receiver is not None:
        receiver.close()
    print("motors stopped")
