"""Read-only fleet readiness check, apart from stopping connected cars."""
from vsss._ble_client import MecanumBLEClient


def main():
    clients = []
    try:
        for number in range(1, 5):
            client = MecanumBLEClient(device_name='car%d' % number)
            clients.append(client)
            try:
                client.connect()
                client.wait_until_connected(12)
                client.stop()
                print(number, client.read_status(), flush=True)
            except Exception as error:
                print(number, type(error).__name__, str(error), flush=True)
    finally:
        for client in clients:
            client.disconnect()


if __name__ == '__main__':
    main()
