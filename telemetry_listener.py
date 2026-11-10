"""Receive and display position telemetry broadcast by main.py."""

import argparse
import json
import socket
import time


DEFAULT_PORT = 45888


def format_packet(message, sender, received_at):
    """Return a compact, readable description of one telemetry packet."""
    timestamp = message.get("timestamp")
    if isinstance(timestamp, (int, float)):
        age_ms = max(0.0, (received_at - timestamp) * 1000)
        age = f" age={age_ms:.1f}ms"
    else:
        age = ""

    board = "yes" if message.get("board_detected") else "no"
    markers = message.get("markers")
    if not isinstance(markers, list):
        raise ValueError("'markers' is not a list")

    if not markers:
        positions = "no markers"
    else:
        formatted = []
        for marker in markers:
            try:
                formatted.append(
                    f"ID {int(marker['id'])}: "
                    f"x={float(marker['x']):+.3f}m, "
                    f"y={float(marker['y']):+.3f}m, "
                    f"angle={float(marker['angle']):+.1f}deg"
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid marker: {marker!r}") from error
        positions = " | ".join(formatted)

    balls = " | ".join(f"ball x={b['x']:+.3f}m y={b['y']:+.3f}m" for b in message.get("balls", []))
    return f"{sender[0]} board={board}{age} | {positions} | {balls or 'no balls'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="local address to listen on (default: all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"telemetry UDP port (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()

    received = 0
    invalid = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            udp.bind((args.bind, args.port))
        except PermissionError as error:
            raise SystemExit(
                f"Cannot listen on UDP port {args.port}: Windows denied the bind. "
                "Another process may already own that port. Choose a free port, "
                f"for example --port 5050, and give main.py the same --port value."
            ) from error
        print(f"Listening for telemetry on {args.bind}:{args.port}. Ctrl+C to stop.")

        try:
            while True:
                data, sender = udp.recvfrom(65535)
                received_at = time.time()
                try:
                    message = json.loads(data)
                    if not isinstance(message, dict):
                        raise ValueError("packet is not a JSON object")
                    print(format_packet(message, sender, received_at), flush=True)
                    received += 1
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    invalid += 1
                    print(f"Invalid packet from {sender[0]}: {error}", flush=True)
        except KeyboardInterrupt:
            pass

    print(f"\nReceived {received} valid packet(s); {invalid} invalid packet(s).")


if __name__ == "__main__":
    main()
