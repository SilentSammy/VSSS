"""Non-blocking Wi-Fi and pose receiver recovery for the robot main loop."""
import time


class TelemetryRecovery:
    def __init__(self, wlan, settings, receiver_factory, retry_ms=10000):
        self.wlan = wlan
        self.settings = settings
        self.receiver_factory = receiver_factory
        self.retry_ms = retry_ms
        self.receiver = None
        self.next_retry = time.ticks_ms()
        self.next_receiver_retry = time.ticks_ms()

    def close(self):
        if self.receiver is not None:
            try:
                self.receiver.close()
            finally:
                self.receiver = None

    def reconnect(self):
        """Drop the old socket and association; poll() starts a new attempt."""
        self.close()
        try:
            self.wlan.disconnect()
        except OSError:
            pass
        self.next_retry = time.ticks_ms()
        self.next_receiver_retry = self.next_retry

    def poll(self):
        now = time.ticks_ms()
        if not self.wlan.isconnected():
            self.close()
            if time.ticks_diff(now, self.next_retry) >= 0:
                self.next_retry = time.ticks_add(now, self.retry_ms)
                try:
                    self.wlan.active(True)
                    self.wlan.connect(self.settings['wifi_ssid'], self.settings['wifi_password'])
                except OSError as error:
                    print('WiFi reconnect failed:', error)
            return
        if self.receiver is None and time.ticks_diff(now, self.next_receiver_retry) >= 0:
            self.next_receiver_retry = time.ticks_add(now, self.retry_ms)
            try:
                self.receiver = self.receiver_factory(
                    self.settings['car_id'], server=self.settings['telemetry_host'],
                    port=self.settings['telemetry_port'], subscribe=True)
            except OSError as error:
                print('Telemetry receiver creation failed:', error)
        if self.receiver is not None:
            self.receiver.poll()
