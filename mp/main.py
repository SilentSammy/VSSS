"""Car firmware: BLE manual/waypoint commands with an onboard UDP pose loop."""
import math
import network
import struct
import time

from axis_trim import AxisTrim
from mecanum import MecanumCar
from onboard_controller import CONTROLLER_KEYS, OnboardController
from pose_receiver import PoseReceiver
from robot_config import RobotConfig, validate
from telemetry_recovery import TelemetryRecovery

VELOCITY_UUID = "12345678-1234-5678-1234-56789abcdef1"
LEASE_UUID = "12345678-1234-5678-1234-56789abcdef2"
TRIM_MATRIX_UUID = "12345678-1234-5678-1234-56789abcdef3"
WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"
CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef7"
CONTROLLER_UUID = "12345678-1234-5678-1234-56789abcdef8"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"

MODE_IDLE = 0
MODE_MANUAL = 1
MODE_POSITION = 2
MODE_REACHED = 3
MODE_FAULT = 4
FAULT_NONE = 0
FAULT_TELEMETRY = 1
FAULT_TIMEOUT = 2
WAYPOINT_TIMEOUT_MS = 30000


def connect_wifi(ssid, password, timeout_ms=20000):
    wlan = network.WLAN(network.STA_IF)
    if wlan.isconnected():
        return wlan
    wlan.active(False)
    time.sleep_ms(500)
    wlan.active(True)
    wlan.config(txpower=5, reconnects=3)
    wlan.connect(ssid, password)
    deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
    while not wlan.isconnected() and time.ticks_diff(deadline, time.ticks_ms()) > 0:
        time.sleep_ms(50)
    return wlan


def controller_payload(parameters):
    return struct.pack("<11f", *(parameters[key] for key in CONTROLLER_KEYS))


def main():
    configuration = RobotConfig()
    settings = configuration.data
    car = MecanumCar()
    car.stop()
    wlan = connect_wifi(settings["wifi_ssid"], settings["wifi_password"])
    telemetry = TelemetryRecovery(wlan, settings, PoseReceiver)

    # Importing ble_server activates BLE, deliberately after WiFi association.
    import ble_server
    state = {
        "mode": MODE_IDLE, "fault": FAULT_NONE, "velocity": (0., 0., 0.),
        "waypoint": None, "deadline": None, "stop": False,
        "control_frame": None, "control_command": (0., 0., 0.),
        "save_trim": False, "save_controller": False,
        "reconnect_at": None, "restart_at": None,
    }

    def trim_changed(_matrix):
        state["save_trim"] = True

    trim = AxisTrim(settings["trim_matrix"], trim_changed)
    controller = OnboardController(**settings["controller"])

    def stop_requested():
        state["stop"] = True
        car.stop()

    def on_velocity(data):
        if state["reconnect_at"] is not None or state["restart_at"] is not None:
            return
        if len(data) == 3:
            state["velocity"] = tuple(ble_server.to_bipolar(value) for value in data)
            state["waypoint"] = None
            state["control_frame"] = None
            state["control_command"] = (0., 0., 0.)
            state["mode"] = MODE_MANUAL if any(data) else MODE_IDLE
            state["fault"] = FAULT_NONE
            controller.clear_target()

    def on_waypoint(data):
        if state["reconnect_at"] is not None or state["restart_at"] is not None:
            return struct.pack("<fff", *(state["waypoint"] or (0., 0., 0.)))
        if len(data) == 12:
            x, y, angle = struct.unpack("<fff", data)
            if all(math.isfinite(v) for v in (x, y, angle)) and -0.8 <= x <= 0.8 and -1.5 <= y <= 1.5:
                state["waypoint"] = (x, y, angle)
                state["control_frame"] = None
                state["control_command"] = (0., 0., 0.)
                # A new BLE waypoint replaces the active target immediately.
                # The main loop installs it on the next iteration.
                controller.clear_target()
                state["mode"] = MODE_POSITION
                state["fault"] = FAULT_NONE
                return data
        return struct.pack("<fff", *(state["waypoint"] or (0., 0., 0.)))

    def on_control(data):
        if data:
            command = data[0]
            if command == 0:
                stop_requested()
            elif command == 1:
                state["save_trim"] = state["save_controller"] = True
            elif command in (2, 3):
                stop_requested()
                if command == 2:
                    state["reconnect_at"] = time.ticks_add(time.ticks_ms(), 350)
                else:
                    state["restart_at"] = time.ticks_add(time.ticks_ms(), 750)
                return bytes((0xa5, command))
        return bytes((state["mode"], state["fault"]))

    def on_controller(data):
        if len(data) == 44:
            values = struct.unpack("<11f", data)
            candidate = dict(zip(CONTROLLER_KEYS, values))
            try:
                test = dict(settings)
                test["controller"] = candidate
                validate(test)
                settings["controller"] = candidate
                state["save_controller"] = True
                return controller_payload(candidate)
            except (ValueError, TypeError):
                pass
        return controller_payload(settings["controller"])

    ble_server.DEVICE_NAME = "car%d" % settings["car_id"]
    ble_server.disconnect_callback = stop_requested
    ble_server.control_callbacks = {
        VELOCITY_UUID: on_velocity,
        LEASE_UUID: lambda data: bytes((state["mode"],)),
        TRIM_MATRIX_UUID: trim.write,
        WAYPOINT_UUID: on_waypoint,
        CONTROL_UUID: on_control,
        CONTROLLER_UUID: on_controller,
        STATUS_UUID: lambda data: None,
    }
    ble_server.initial_values = {
        VELOCITY_UUID: bytes(3), LEASE_UUID: bytes(1),
        TRIM_MATRIX_UUID: trim.read(), WAYPOINT_UUID: bytes(12),
        CONTROL_UUID: bytes(2), CONTROLLER_UUID: controller_payload(settings["controller"]),
        STATUS_UUID: bytes(20),
    }

    try:
        ble_server.start()
        print("car%d ready; WiFi %s; control onboard" % (
            settings["car_id"], wlan.ifconfig()[0] if wlan.isconnected() else "unavailable"))
        last_status = 0
        while True:
            now = time.ticks_ms()
            ble_server.poll()

            if state["stop"]:
                state["stop"] = False
                state["mode"], state["fault"] = MODE_IDLE, FAULT_NONE
                state["velocity"], state["waypoint"] = (0., 0., 0.), None
                state["control_frame"] = None
                state["control_command"] = (0., 0., 0.)
                state["deadline"] = None
                controller.clear_target()
                car.stop()

            if state["restart_at"] is not None and time.ticks_diff(now, state["restart_at"]) >= 0:
                car.stop()
                import machine
                machine.reset()

            if state["reconnect_at"] is not None and time.ticks_diff(now, state["reconnect_at"]) >= 0:
                state["reconnect_at"] = None
                telemetry.reconnect()
            telemetry.poll()
            receiver = telemetry.receiver

            if state["save_trim"]:
                state["save_trim"] = False
                configuration.set_trim(trim.matrix)
            if state["save_controller"]:
                state["save_controller"] = False
                configuration.set_controller(settings["controller"])
                controller = OnboardController(**settings["controller"])

            if state["mode"] == MODE_MANUAL:
                car.set_velocity(*trim.apply(state["velocity"]))
            elif state["mode"] == MODE_IDLE:
                # A zero velocity write enters idle.  Explicitly release the
                # motor channels; otherwise their previous PWM values remain
                # active until another nonzero command replaces them.
                car.stop()
            elif state["mode"] == MODE_POSITION:
                if controller.target is None and state["waypoint"] is not None:
                    x, y, angle = state["waypoint"]
                    controller.set_target(x, y, math.radians(angle))
                    state["control_frame"] = None
                    state["control_command"] = (0., 0., 0.)
                    state["deadline"] = time.ticks_add(now, WAYPOINT_TIMEOUT_MS)
                if receiver is None or not wlan.isconnected() or not receiver.fresh:
                    car.stop()
                    state["mode"], state["fault"] = MODE_FAULT, FAULT_TELEMETRY
                elif time.ticks_diff(now, state["deadline"]) >= 0:
                    car.stop()
                    state["mode"], state["fault"] = MODE_FAULT, FAULT_TIMEOUT
                else:
                    # Camera samples arrive much more slowly than this loop.
                    # Re-estimating velocity from the same pose at 100 Hz makes
                    # the next real displacement look as if it occurred in
                    # 10 ms, producing a large, noisy predictive correction.
                    frame = (receiver.frame_sequence if receiver.frame_sequence is not None
                             else receiver.packets)
                    if frame != state["control_frame"]:
                        state["control_frame"] = frame
                        command = controller.update(receiver.pose, receiver.received_ms)
                        state["control_command"] = trim.apply(command)
                        car.set_velocity(*state["control_command"])
                        if controller.reached(receiver.received_ms):
                            car.stop()
                            state["control_command"] = (0., 0., 0.)
                            state["mode"], state["fault"] = MODE_REACHED, FAULT_NONE

            if time.ticks_diff(now, last_status) >= 100:
                last_status = now
                pose = receiver.pose if receiver is not None and receiver.fresh else (0., 0., 0.)
                flags = ((1 if wlan.isconnected() else 0) |
                         (2 if receiver is not None and receiver.fresh else 0))
                packets = receiver.packets if receiver is not None else 0
                ble_server.set_value(STATUS_UUID, struct.pack(
                    "<BBBBfffI", state["mode"], flags, state["fault"], 0,
                    pose[0], pose[1], pose[2], packets))
            time.sleep_ms(10)
    except KeyboardInterrupt:
        pass
    finally:
        car.stop()
        telemetry.close()
        ble_server.stop()


if __name__ == "__main__":
    main()
