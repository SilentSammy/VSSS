# Hotspot-host UDP comparison

Tested November 3, 2026. ESP32-C3 on COM17, MicroPython 1.29.0,
GPIO8 LED commands with JSON acknowledgements on UDP port 5002.
ESP32 address: 192.168.12.253. BLE disabled, WiFi power management disabled,
transmit power 8.5 dBm. Initial RSSI: -63 dBm.

| Sender | Sent / acknowledged | Missing replies | Median RTT | p95 RTT | p99 RTT | Max RTT |
| --- | --- | --- | --- | --- | --- | --- |
| Hotspot Ubuntu host, 192.168.12.1 | 1500 / 1494 | 6 (0.4%) | 5.28 ms | 14.06 ms | 25.16 ms | 414.70 ms |
| Windows PC, 192.168.12.195 | 500 / 494 | 6 (1.2%) | 8.10 ms | 23.50 ms | 41.01 ms | 389.76 ms |

Three sequential 500-command trials on the hotspot, followed by one trial
on Windows, with identical firmware and client code. Each trial targets
20 ms between sends. Each command waits for its matching acknowledgement,
with a 500 ms deadline, so slow replies and timeouts reduce the actual send
rate. No commands are retransmitted. The acknowledgement follows the LED
update. ESP32 processing median was approximately 0.57 ms.

RTT statistics include successful replies only. A missing reply means no
matching acknowledgement within 500 ms; this does not distinguish a lost
command, lost reply, or late reply. No division by two or clock synchronization
is used: these are round-trip measurements.

The hotspot sender improved median and tail latency in this run, but occasional
large delays and missed replies remain. Trials were sequential and unequal in
number, so this does not isolate the effects of OS, radio, interference, and
host scheduling. The third hotspot trial received all 500 replies, with a
5.23 ms median and 47.15 ms maximum.

SSH used Tailscale to launch the remote client. Measured UDP packets traveled
directly from ap0 (192.168.12.1) to the ESP32, outside the SSH/Tailscale path.
Raw results: `hotspot_results.json` and `local_results.json`.

The temporary remote directory was removed after results were downloaded.
The ESP32 server was stopped, GPIO8 left off, and WiFi disabled; COM17 released.
Device boot files were not changed.
