# Dual motor driver test - PWM on direction pins, enable always HIGH
from machine import Pin, PWM
import time

# Motor 1 pin assignments (PWM control)
IN1A_PIN = 5  # 1A - PWM for direction/speed
IN2A_PIN = 6  # 2A - PWM for direction/speed

# Motor 2 pin assignments (PWM control)
IN3A_PIN = 7  # 3A - PWM for direction/speed
IN4A_PIN = 8  # 4A - PWM for direction/speed

# Setup PWM pins
# Motor 1
in1a = PWM(Pin(IN1A_PIN), freq=1000)
in2a = PWM(Pin(IN2A_PIN), freq=1000)

# Motor 2
in3a = PWM(Pin(IN3A_PIN), freq=1000)
in4a = PWM(Pin(IN4A_PIN), freq=1000)

# Note: Wire enable pins (1,2EN and 3,4EN) directly to 3.3V

# Test sequence
print("Dual motor PWM test starting...")

# Test 1: Motor 1 only - forward at 30% speed
print("Test 1: Motor 1 forward 30%...")
in1a.duty_u16(19660)  # 30% speed
in2a.duty_u16(0)      # OFF
in3a.duty_u16(0)      # Motor 2 off
in4a.duty_u16(0)
time.sleep(2)

# Stop
print("Stop...")
in1a.duty_u16(0)
in2a.duty_u16(0)
time.sleep(1)

# Test 2: Motor 2 only - forward at 30% speed
print("Test 2: Motor 2 forward 30%...")
in1a.duty_u16(0)      # Motor 1 off
in2a.duty_u16(0)
in3a.duty_u16(19660)  # 30% speed
in4a.duty_u16(0)      # OFF
time.sleep(2)

# Stop
print("Stop...")
in3a.duty_u16(0)
in4a.duty_u16(0)
time.sleep(1)

# Test 3: Both motors forward at 30% speed
print("Test 3: Both motors forward 30%...")
in1a.duty_u16(19660)  # 30% speed
in2a.duty_u16(0)      # OFF
in3a.duty_u16(19660)  # 30% speed
in4a.duty_u16(0)      # OFF
time.sleep(2)

# Stop
print("Stop...")
in1a.duty_u16(0)
in2a.duty_u16(0)
in3a.duty_u16(0)
in4a.duty_u16(0)
time.sleep(1)

# Test 4: Motor 1 reverse at 30% speed
print("Test 4: Motor 1 reverse 30%...")
in1a.duty_u16(0)      # OFF
in2a.duty_u16(19660)  # 30% speed
in3a.duty_u16(0)      # Motor 2 off
in4a.duty_u16(0)
time.sleep(2)

# Stop
print("Stop...")
in1a.duty_u16(0)
in2a.duty_u16(0)
time.sleep(1)

# Test 5: Motor 2 reverse at 30% speed
print("Test 5: Motor 2 reverse 30%...")
in1a.duty_u16(0)      # Motor 1 off
in2a.duty_u16(0)
in3a.duty_u16(0)      # OFF
in4a.duty_u16(19660)  # 30% speed
time.sleep(2)

# Stop
print("Stop...")
in3a.duty_u16(0)
in4a.duty_u16(0)
time.sleep(1)

# Test 6: Both motors reverse at 30% speed
print("Test 6: Both motors reverse 30%...")
in1a.duty_u16(0)      # OFF
in2a.duty_u16(19660)  # 30% speed
in3a.duty_u16(0)      # OFF
in4a.duty_u16(19660)  # 30% speed
time.sleep(2)

# Stop
print("Stop...")
in1a.duty_u16(0)
in2a.duty_u16(0)
in3a.duty_u16(0)
in4a.duty_u16(0)

print("Test complete!")
