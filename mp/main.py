# Dual motor driver test - PWM on direction pins, enable always HIGH
from machine import Pin, PWM
import time
import ble_server
from mecanum import MecanumCar

if __name__ == "__main__":
    print("Mecanum car BLE control starting...")
    
    # Create mecanum car
    car = MecanumCar()
    
    # Define BLE callback for combined velocity
    def on_velocity(data):
        """Handle velocity update (3 bytes: x, y, w)"""
        if len(data) >= 3:
            car.x = ble_server.to_bipolar(data[0])
            car.y = ble_server.to_bipolar(data[1])
            car.w = ble_server.to_bipolar(data[2])
    
    # Register callback with single UUID
    ble_server.control_callbacks = {
        '12345678-1234-5678-1234-56789abcdef1': on_velocity,  # Combined velocity (x, y, w)
    }
    
    # Start BLE server
    ble_server.start()
    
    print("\nBLE control active. Use BLE client to control the car.")
    print("Characteristics:")
    print("  Velocity (x,y,w):    ...def1")
    
