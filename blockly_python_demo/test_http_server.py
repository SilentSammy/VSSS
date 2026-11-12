"""Browser-test fixture only. Production server.py always uses the real Car."""
import sys
from live_runtime import LiveRunner
from server import serve
from test_live_runtime import FakeCar

if __name__ == '__main__':
    serve(int(sys.argv[1]), LiveRunner(FakeCar))
