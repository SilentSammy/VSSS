"""RAM-only GPIO8 UDP round-trip test. Credentials passed by the launcher."""
import bluetooth
import network
import socket
import time
import json
from machine import Pin


def run(ssid, password):
    bluetooth.BLE().active(False)
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.config(txpower=8.5, reconnects=2)
    if hasattr(network.WLAN, 'PM_NONE'):
        wlan.config(pm=network.WLAN.PM_NONE)
    wlan.connect(ssid, password)
    started = time.ticks_ms()
    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), started) > 30000:
            raise OSError('WiFi connection timed out')
        time.sleep_ms(100)
    led = Pin(8, Pin.OUT, value=1)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(('0.0.0.0', 5002))
    print('READY', wlan.ifconfig()[0], 'RSSI', wlan.status('rssi'))
    try:
        while True:
            data, peer = udp.recvfrom(256)
            received = time.ticks_us()
            try:
                command = json.loads(data)
                state, seq = command['led'], command['seq']
                if state not in (0, 1):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            led.value(1 - int(state))
            udp.sendto(json.dumps({'seq': seq, 'led': state,
                                  'processing_us': time.ticks_diff(time.ticks_us(), received)}).encode(), peer)
    finally:
        led.value(1)
        udp.close()
