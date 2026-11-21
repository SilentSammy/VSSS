"""Car firmware: BLE manual/waypoint commands with an onboard UDP pose loop."""
import math
import network
import struct
import time
try:
    import ujson as json
except ImportError:
    import json

from axis_trim import AxisTrim
from mecanum import MecanumCar
from onboard_controller import CONTROLLER_KEYS, OnboardController
from pose_receiver import PoseReceiver
from robot_config import RobotConfig, validate
from telemetry_recovery import TelemetryFreshness, TelemetryRecovery, low_latency_wifi

VELOCITY_UUID = "12345678-1234-5678-1234-56789abcdef1"
LEASE_UUID = "12345678-1234-5678-1234-56789abcdef2"
TRIM_MATRIX_UUID = "12345678-1234-5678-1234-56789abcdef3"
WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"
CONTROL_UUID = "12345678-1234-5678-1234-56789abcdef7"
CONTROLLER_UUID = "12345678-1234-5678-1234-56789abcdef8"
STATUS_UUID = "12345678-1234-5678-1234-56789abcdef9"
CONFIG_UUID = "12345678-1234-5678-1234-56789abcdefa"
CONFIG_MAX_BYTES = 2048

MODE_IDLE = 0
MODE_MANUAL = 1
MODE_POSITION = 2
MODE_REACHED = 3
MODE_FAULT = 4
FAULT_NONE = 0
FAULT_TELEMETRY = 1
FAULT_TIMEOUT = 2

# Keep the car object available from the REPL after main() is interrupted.
# It is assigned during startup and remains usable for diagnostics/manual tests.
car = None

def connect_wifi(ssid, password, timeout_ms=20000):
    wlan = network.WLAN(network.STA_IF)
    if wlan.isconnected():
        low_latency_wifi(wlan)
        return wlan
    wlan.active(False)
    time.sleep_ms(500)
    wlan.active(True)
    low_latency_wifi(wlan)
    wlan.config(txpower=5, reconnects=3)
    wlan.connect(ssid, password)
    deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
    while not wlan.isconnected() and time.ticks_diff(deadline, time.ticks_ms()) > 0:
        time.sleep_ms(50)
    return wlan


def controller_payload(parameters):
    return struct.pack("<%df" % len(CONTROLLER_KEYS),
                       *(parameters[key] for key in CONTROLLER_KEYS))


def main():
    global car
    configuration = RobotConfig()
    settings = configuration.data
    car = MecanumCar(motor_pins=settings["motor_pins"],
                     wheel_gains=settings.get("wheel_gains"))
    car.wheel_reverse_gains = tuple(settings.get("wheel_reverse_gains", (1., 1., 1., 1.)))
    car.stop()
    try:
        wlan = connect_wifi(settings["wifi_ssid"], settings["wifi_password"])
        telemetry = TelemetryRecovery(wlan, settings, PoseReceiver)
        # Importing ble_server activates BLE, deliberately after WiFi association.
        import ble_server
    except BaseException:
        car.stop()
        raise
    nav_log = None
    state = {
        "mode": MODE_IDLE, "fault": FAULT_NONE, "velocity": (0., 0., 0.),
        "waypoint": None, "deadline": None, "stop": False,
        "control_frame": None, "control_command": (0., 0., 0.),
        "control_raw": (0., 0., 0.), "control_dt_ms": None, "waypoint_id": 0,
        "save_trim": False, "save_controller": False,
        "config_update": None, "config_rx": None, "config_rx_size": 0,
        "reconnect_at": None, "restart_at": None,
    }

    def trim_changed(_matrix):
        state["save_trim"] = True

    trim = AxisTrim(settings["trim_matrix"], trim_changed, settings["mixed_trim"])
    controller = OnboardController(**settings["controller"])
    freshness = TelemetryFreshness(settings["controller"]["telemetry_recovery_ms"])

    def stop_requested():
        state["stop"] = True
        car.stop()

    def config_payload(candidate=None):
        """Return the complete canonical file, including credentials.

        This characteristic is intentionally administrative: a BLE peer that
        can connect to the car can read and replace all persisted settings.
        """
        if candidate is None:
            candidate = dict(settings)
            candidate["trim_matrix"] = list(trim.matrix)
            candidate["controller"] = dict(settings["controller"])
        return json.dumps(validate(candidate)).encode()

    def on_config(data):
        """Accept a full JSON file or a numbered chunked JSON transfer.

        Long values exceed conservative BLE write limits on some hosts.  A
        chunk starts with 0x01, then little-endian offset and total length.
        Chunks must arrive contiguously; the response always remains the last
        complete configuration so ordinary GATT reads are stable.
        """
        payload = bytes(data)
        try:
            if payload[:1] == b"\x01":
                if len(payload) < 5:
                    raise ValueError("short config chunk")
                offset, total = struct.unpack("<HH", payload[1:5])
                chunk = payload[5:]
                if not 1 <= total <= CONFIG_MAX_BYTES or offset + len(chunk) > total:
                    raise ValueError("invalid config chunk")
                if offset == 0:
                    state["config_rx"] = bytearray(total)
                    state["config_rx_size"] = 0
                if state["config_rx"] is None or len(state["config_rx"]) != total or offset != state["config_rx_size"]:
                    raise ValueError("out-of-order config chunk")
                state["config_rx"][offset:offset + len(chunk)] = chunk
                state["config_rx_size"] += len(chunk)
                if state["config_rx_size"] != total:
                    return config_payload()
                payload = bytes(state["config_rx"])
                state["config_rx"] = None
                state["config_rx_size"] = 0
            candidate = validate(json.loads(payload.decode()))
            state["config_update"] = candidate
            return config_payload(candidate)
        except (ValueError, TypeError, UnicodeError):
            state["config_rx"] = None
            state["config_rx_size"] = 0
            return config_payload()

    def on_velocity(data):
        if state["reconnect_at"] is not None or state["restart_at"] is not None:
            return
        if len(data) == 3:
            state["velocity"] = tuple(ble_server.to_bipolar(value) for value in data)
            state["waypoint"] = None
            state["control_frame"] = None
            state["control_command"] = (0., 0., 0.)
            state["control_raw"], state["control_dt_ms"] = (0., 0., 0.), None
            state["mode"] = MODE_MANUAL if any(data) else MODE_IDLE
            state["fault"] = FAULT_NONE
            freshness.reset()
            controller.clear_target()

    def on_waypoint(data):
        if state["reconnect_at"] is not None or state["restart_at"] is not None:
            return struct.pack("<fff", *(state["waypoint"] or (0., 0., 0.)))
        if len(data) == 12:
            x, y, angle = struct.unpack("<fff", data)
            if all(math.isfinite(v) for v in (x, y, angle)) and -0.8 <= x <= 0.8 and -1.5 <= y <= 1.5:
                state["waypoint"] = (x, y, angle)
                state["waypoint_id"] += 1
                state["control_frame"] = None
                state["control_command"] = (0., 0., 0.)
                state["control_raw"], state["control_dt_ms"] = (0., 0., 0.), None
                # A new BLE waypoint replaces the active target immediately.
                # The main loop installs it on the next iteration.
                controller.clear_target()
                state["mode"] = MODE_POSITION
                state["fault"] = FAULT_NONE
                freshness.reset()
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
        if len(data) == 4 * len(CONTROLLER_KEYS):
            values = struct.unpack("<%df" % len(CONTROLLER_KEYS), data)
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
        CONFIG_UUID: on_config,
    }
    ble_server.initial_values = {
        VELOCITY_UUID: bytes(3), LEASE_UUID: bytes(1),
        TRIM_MATRIX_UUID: trim.read(), WAYPOINT_UUID: bytes(12),
        CONTROL_UUID: bytes(2), CONTROLLER_UUID: controller_payload(settings["controller"]),
        STATUS_UUID: bytes(20), CONFIG_UUID: config_payload(),
    }
    ble_server.buffer_sizes = {CONFIG_UUID: CONFIG_MAX_BYTES}

    try:
        ble_server.start()
        # Fully start Wi-Fi and GATT before allocating optional diagnostic RAM.
        from navigation_log import NavigationLog
        nav_log = NavigationLog(settings)
        nav_log.event('navigation_policy', wifi_pm=wlan.config('pm'),
                      freshness='frame_age_plus_time_since_unique_frame', stale_ms=500,
                      settle_speed=controller.settle_speed,
                      settle_yaw_rad_s=controller.settle_yaw_rate)
        nav_log.flush()  # Startup is stopped; no flash writes during motion.
        import gc
        gc.collect()  # Release startup serialization/GATT temporaries before RX.
        print("car%d ready; WiFi %s; control onboard" % (
            settings["car_id"], wlan.ifconfig()[0] if wlan.isconnected() else "unavailable"))
        last_status = 0
        previous_loop = time.ticks_ms()
        while True:
            now = time.ticks_ms()
            loop_ms = time.ticks_diff(now, previous_loop)
            previous_loop = now
            ble_server.poll()

            if state["stop"]:
                state["stop"] = False
                state["mode"], state["fault"] = MODE_IDLE, FAULT_NONE
                state["velocity"], state["waypoint"] = (0., 0., 0.), None
                state["control_frame"] = None
                state["control_command"] = (0., 0., 0.)
                state["control_raw"], state["control_dt_ms"] = (0., 0., 0.), None
                state["deadline"] = None
                freshness.reset()
                controller.clear_target()
                car.stop()

            if state["restart_at"] is not None and time.ticks_diff(now, state["restart_at"]) >= 0:
                car.stop()
                nav_log.event('restart')
                nav_log.flush()
                import machine
                machine.reset()

            if state["reconnect_at"] is not None and time.ticks_diff(now, state["reconnect_at"]) >= 0:
                state["reconnect_at"] = None
                nav_log.event('telemetry_reconnect')
                telemetry.reconnect()
            telemetry.poll()
            receiver = telemetry.receiver

            if state["save_trim"]:
                state["save_trim"] = False
                configuration.set_trim(trim.matrix)
                ble_server.set_value(CONFIG_UUID, config_payload())
                nav_log.configuration(trim.matrix, settings["controller"])
            if state["save_controller"]:
                state["save_controller"] = False
                configuration.set_controller(settings["controller"])
                ble_server.set_value(CONFIG_UUID, config_payload())
                controller = OnboardController(**settings["controller"])
                freshness = TelemetryFreshness(settings["controller"]["telemetry_recovery_ms"])
                nav_log.configuration(trim.matrix, settings["controller"])
            if state["config_update"] is not None:
                # Never write flash or rebuild hardware from the BLE IRQ.
                # Rebooting makes every setting (identity, Wi-Fi and pins) take
                # effect atomically after the persisted configuration is valid.
                candidate = state["config_update"]
                state["config_update"] = None
                car.stop()
                configuration.replace(candidate)
                settings.clear()
                settings.update(configuration.data)
                state["save_trim"] = state["save_controller"] = False
                nav_log.event('configuration_update', car_id=settings['car_id'],
                              motor_pins=settings['motor_pins'])
                nav_log.flush()
                state["restart_at"] = time.ticks_add(time.ticks_ms(), 750)

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
                    state["deadline"] = time.ticks_add(
                        now, int(settings["controller"]["waypoint_timeout_ms"]))
                feedback_fresh = (receiver is not None and wlan.isconnected() and receiver.fresh)
                feedback_state, outage_ms = freshness.update(feedback_fresh, now)
                if feedback_state in ("pause", "wait", "fault"):
                    car.stop()
                    state["control_command"] = (0., 0., 0.)
                    if feedback_state == "pause":
                        nav_log.event('telemetry_pause',
                                      age_ms=receiver.age_ms if receiver else None,
                                      frame_sequence=receiver.frame_sequence if receiver else None,
                                      packets=receiver.packets if receiver else 0)
                    if feedback_state == "fault":
                        state["mode"], state["fault"] = MODE_FAULT, FAULT_TELEMETRY
                        try:
                            fault_rssi = int(wlan.status('rssi')) if wlan.isconnected() else None
                        except (OSError, ValueError, TypeError):
                            fault_rssi = None
                        nav_log.event('telemetry_fault',
                                      age_ms=receiver.age_ms if receiver else None,
                                      rx_age_ms=(time.ticks_diff(now, receiver.received_ms)
                                                 if receiver and receiver.received_ms is not None else None),
                                      frame_age_ms=receiver.frame_age_ms if receiver else None,
                                      frame_sequence=receiver.frame_sequence if receiver else None,
                                      packets=receiver.packets if receiver else 0,
                                      sequence_gap=receiver.sequence_gap if receiver else None,
                                      receive_gap_ms=receiver.receive_gap_ms if receiver else None,
                                      wifi_rssi=fault_rssi,
                                      outage_ms=outage_ms,
                                      memory_errors=receiver.memory_errors if receiver else 0)
                elif feedback_state == "recover":
                    # Do not estimate velocity across a telemetry gap. Resume
                    # the retained target from the newly fresh pose instead.
                    controller.reset()
                    state["control_frame"] = None
                    state["control_raw"], state["control_dt_ms"] = (0., 0., 0.), None
                    nav_log.event('telemetry_recovered', outage_ms=outage_ms,
                                  frame_sequence=receiver.frame_sequence,
                                  packets=receiver.packets)
                elif time.ticks_diff(now, state["deadline"]) >= 0:
                    car.stop()
                    state["mode"], state["fault"] = MODE_FAULT, FAULT_TIMEOUT
                else:
                    # Camera samples arrive much more slowly than this loop.
                    # Re-estimating velocity from the same pose at 100 Hz makes
                    # the next real displacement look as if it occurred in
                    # 10 ms, producing a large, noisy predictive correction.
                    frame = (receiver.stream_id, receiver.frame_sequence if receiver.frame_sequence is not None
                             else receiver.packets)
                    if frame != state["control_frame"]:
                        state["control_frame"] = frame
                        state["control_dt_ms"] = (time.ticks_diff(receiver.received_ms, controller.previous[0])
                                                   if controller.previous is not None else None)
                        command = controller.update(receiver.pose, receiver.received_ms)
                        state["control_raw"] = command
                        state["control_command"] = trim.apply(command)
                        car.set_velocity(*state["control_command"])
                        if controller.reached(receiver.received_ms):
                            car.stop()
                            state["control_command"] = (0., 0., 0.)
                            state["mode"], state["fault"] = MODE_REACHED, FAULT_NONE

            if nav_log.sample(now, state, receiver, controller, car, wlan, loop_ms):
                # Stopped-state flash writes can take longer than the feedback
                # timeout. Drain queued telemetry before publishing status;
                # genuinely stale frames still fail the unchanged age check.
                telemetry.poll()
                receiver = telemetry.receiver
                now = time.ticks_ms()
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
    except Exception as error:
        if nav_log is not None:
            nav_log.event('exception', error=repr(error))
        raise
    finally:
        car.stop()
        if nav_log is not None:
            nav_log.event('shutdown', mode=state['mode'], fault=state['fault'])
            nav_log.close()
        telemetry.close()
        ble_server.stop()


if __name__ == "__main__":
    main()
