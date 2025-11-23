# Minimal motor driver test for SN754410NE - Digital only
from machine import Pin
import time

# Pin assignments
EN_PIN = 4   # 3,4EN - Enable (digital HIGH/LOW)
IN1_PIN = 3  # 3A - Direction pin 1
IN2_PIN = 2  # 4A - Direction pin 2

# Setup pins - all digital outputs
enable = Pin(EN_PIN, Pin.OUT)
in1 = Pin(IN1_PIN, Pin.OUT)
in2 = Pin(IN2_PIN, Pin.OUT)

# Test sequence
print("Motor test starting (digital only)...")

# Test 1: Forward
print("Test 1: Forward full speed...")
enable.value(1)  # Enable ON
in1.value(1)
in2.value(0)
time.sleep(2)

# Stop
print("Stop...")
enable.value(0)  # Enable OFF
time.sleep(1)

# Test 2: Reverse
print("Test 2: Reverse full speed...")
enable.value(1)  # Enable ON
in1.value(0)
in2.value(1)
time.sleep(2)

# Stop
print("Stop...")
enable.value(0)  # Enable OFF
in1.value(0)
in2.value(0)

print("Test complete!")
