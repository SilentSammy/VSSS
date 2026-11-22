"""Browser-test fixture only. Production always uses real vsss devices."""
import sys
from fake_devices import Devices
from server import serve

if __name__ == '__main__':
    serve(int(sys.argv[1]), Devices().runner())
