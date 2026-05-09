# Dual motor driver test - PWM on direction pins, enable always HIGH
from machine import Pin, PWM
import time
import ble_server
from mecanum import MecanumCar

ble_server.DEVICE_NAME = "Eq4"
car = MecanumCar()
mfl = car.motor_fl
mfr = car.motor_fr
mbl = car.motor_bl
mbr = car.motor_br

def test():
    """Test each motor individually: forward, backward, stop."""
    test_power = 0.5
    test_duration = 2
    
    try:
        print("Testing Front-Left motor...")
        car.motor_fl.set_power(test_power)
        time.sleep(test_duration)
        car.motor_fl.set_power(-test_power)
        time.sleep(test_duration)
        car.motor_fl.stop()
        time.sleep(1)
        
        print("Testing Front-Right motor...")
        car.motor_fr.set_power(test_power)
        time.sleep(test_duration)
        car.motor_fr.set_power(-test_power)
        time.sleep(test_duration)
        car.motor_fr.stop()
        time.sleep(1)
        
        print("Testing Back-Left motor...")
        car.motor_bl.set_power(test_power)
        time.sleep(test_duration)
        car.motor_bl.set_power(-test_power)
        time.sleep(test_duration)
        car.motor_bl.stop()
        time.sleep(1)
        
        print("Testing Back-Right motor...")
        car.motor_br.set_power(test_power)
        time.sleep(test_duration)
        car.motor_br.set_power(-test_power)
        time.sleep(test_duration)
        car.motor_br.stop()
        
        print("Per-motor test complete!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        car.stop()

def main():
    print("Mecanum car BLE control starting...")
    
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

if __name__ == "__main__":
    # test()
    main()
