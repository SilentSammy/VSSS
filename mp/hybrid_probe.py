"""COM18 test: run BLE and VSSS UDP telemetry concurrently, without motors."""
import network
import struct
import time

from pose_receiver import PoseReceiver

DEVICE_NAME = "car18-test"
COMMAND_UUID = "12345678-1234-5678-1234-56789abcdef1"
POSE_UUID = "12345678-1234-5678-1234-56789abcdef4"
STATS_UUID = "12345678-1234-5678-1234-56789abcdef5"
WAYPOINT_UUID = "12345678-1234-5678-1234-56789abcdef6"


def connect_wifi(ssid, password, timeout_ms=20000):
    wlan = network.WLAN(network.STA_IF)
    if not wlan.isconnected():
        # Some compact ESP32-C3 boards fail WPA association at the default
        # transmit power even when scanning reports a strong signal.
        wlan.active(False)
        time.sleep_ms(500)
        wlan.active(True)
        wlan.config(txpower=5, reconnects=3)
        wlan.connect(ssid, password)
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
        while not wlan.isconnected():
            if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                raise OSError("WiFi connection timed out: status %s" % wlan.status())
            time.sleep_ms(50)
    return wlan


def main(ssid="embedded", password="12345678", marker_id=1,
         server="255.255.255.255", subscribe=True):
    wlan = connect_wifi(ssid, password)
    # On ESP32-C3, associate WiFi before enabling BLE. Starting BLE during the
    # initial WiFi scan can make the first association fail despite good RSSI.
    import ble_server
    from onboard_controller import OnboardController
    receiver = PoseReceiver(marker_id, server=server, subscribe=subscribe)
    controller = OnboardController()
    last_command = bytes(3)
    last_report = time.ticks_ms()
    last_packets = 0
    compute_us = 0
    compute_samples = 0

    def on_command(data):
        nonlocal last_command
        if len(data) == 3:
            last_command = bytes(data)

    def on_waypoint(data):
        if len(data) == 12:
            x, y, angle_degrees = struct.unpack("<fff", data)
            controller.set_target(x, y, angle_degrees * 3.141592653589793 / 180)

    ble_server.DEVICE_NAME = DEVICE_NAME
    ble_server.disconnect_callback = controller.clear_target
    ble_server.control_callbacks = {
        COMMAND_UUID: on_command,
        POSE_UUID: lambda data: None,
        STATS_UUID: lambda data: None,
        WAYPOINT_UUID: on_waypoint,
    }
    ble_server.initial_values = {
        COMMAND_UUID: last_command,
        POSE_UUID: struct.pack("<fff", 0.0, 0.0, 0.0),
        STATS_UUID: struct.pack("<II", 0, 0),
        WAYPOINT_UUID: struct.pack("<fff", 0.0, 0.0, 0.0),
    }
    try:
        ble_server.start()
        print("WiFi", wlan.ifconfig()[0], "BLE", DEVICE_NAME,
              "telemetry", "unicast" if subscribe else "broadcast")
        while True:
            ble_server.poll()
            if receiver.poll():
                before = time.ticks_us()
                command = controller.update(receiver.pose if receiver.fresh else None,
                                            time.ticks_ms())
                compute_us += time.ticks_diff(time.ticks_us(), before)
                compute_samples += 1
                pose = receiver.pose if receiver.pose is not None else (0.0, 0.0, 0.0)
                ble_server.set_value(POSE_UUID, struct.pack("<fff", *pose), notify=True)
                ble_server.set_value(STATS_UUID, struct.pack(
                    "<II", receiver.packets, receiver.parse_errors))
            now = time.ticks_ms()
            if time.ticks_diff(now, last_report) >= 1000:
                elapsed = max(1, time.ticks_diff(now, last_report))
                rate = (receiver.packets - last_packets) * 1000 / elapsed
                print("udp=%.1fHz packets=%d fresh=%s pose=%s frame_age=%s ble=%d" % (
                    rate, receiver.packets, receiver.fresh, receiver.pose,
                    receiver.frame_age_ms, len(ble_server._connections)))
                if compute_samples:
                    print("controller_us=%.1f command=%s" % (
                        compute_us / compute_samples, controller.command))
                last_report, last_packets = now, receiver.packets
            time.sleep_ms(5)
    finally:
        receiver.close()
        ble_server.stop()


if __name__ == "__main__":
    main()
