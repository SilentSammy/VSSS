# Minimal motor driver test for SN754410NE
from machine import Pin, PWM
import time

# Pin assignments
EN_PIN = 5   # 1,2EN - Enable (PWM for speed control)
IN1_PIN = 6  # 1A - Direction pin 1
IN2_PIN = 7  # 2A - Direction pin 2

# Setup pins
enable = PWM(Pin(EN_PIN), freq=1000)  # 1kHz PWM
in1 = Pin(IN1_PIN, Pin.OUT)
in2 = Pin(IN2_PIN, Pin.OUT)

# Test sequence
print("Motor test starting...")

# Forward
print("Forward...")
in1.value(1)
in2.value(0)
enable.duty(512)  # 50% speed (0-1023 range)
time.sleep(2)

# Stop
print("Stop...")
enable.duty(0)
time.sleep(1)

# Reverse
print("Reverse...")
in1.value(0)
in2.value(1)
enable.duty(512)  # 50% speed
time.sleep(2)

# Stop
print("Stop...")
enable.duty(0)
in1.value(0)
in2.value(0)

print("Test complete!")
