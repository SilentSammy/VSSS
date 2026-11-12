"""Receive pose telemetry by reliable unicast subscription or legacy broadcast."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import socket
import time


DEFAULT_PORT = 5000


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

    balls = message.get('balls')
    if isinstance(balls, list) and balls:
        try:
            positions += ' | ' + ' | '.join(
                f"ball: x={float(ball['x']):+.3f}m, y={float(ball['y']):+.3f}m"
                for ball in balls)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid ball coordinates') from error
    return f"{sender[0]} board={board}{age} | {positions}"


def main(argv=None):
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
        help=f"broadcast data port, or unicast service data port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--server",
        help="vision server IP/hostname; subscribes by unicast instead of listening for broadcast",
    )
    parser.add_argument(
        "--subscription-port",
        type=int,
        help="server subscription port (default: telemetry port + 1)",
    )
    parser.add_argument("--listen-port", type=int,
                        help="local UDP port in unicast mode (default: automatic)")
    parser.add_argument("--duration", type=float, metavar="SECONDS",
                        help="stop after this many seconds (default: until Ctrl+C)")
    parser.add_argument("--output", type=Path, metavar="FILE",
                        help="record valid packets as JSON Lines in a new file")
    parser.add_argument("--quiet", action="store_true", help="suppress individual packet output")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.subscription_port is not None and not 1 <= args.subscription_port <= 65535:
        parser.error('--subscription-port must be between 1 and 65535')
    if args.server and args.subscription_port is None and args.port == 65535:
        parser.error('--subscription-port is required when --port is 65535')
    if args.listen_port is not None and (not args.server or not 0 <= args.listen_port <= 65535):
        parser.error('--listen-port requires --server and must be between 0 and 65535')
    if args.subscription_port is not None and not args.server:
        parser.error('--subscription-port requires --server')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')

    received = 0
    invalid = 0

    with ExitStack() as stack:
        udp = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_port = (args.listen_port or 0) if args.server else args.port
        try:
            udp.bind((args.bind, listen_port))
        except PermissionError as error:
            raise SystemExit(
                f"Cannot listen on UDP port {listen_port}: Windows denied the bind. "
                "Another process may already own that port. Choose a free local port."
            ) from error
        recording = stack.enter_context(args.output.open('x', encoding='utf-8')) if args.output else None
        udp.settimeout(0.2)
        subscription_port = args.subscription_port or args.port + 1
        subscription = (args.server, subscription_port) if args.server else None
        next_renewal = 0.0
        if subscription:
            print(f"Subscribing to {args.server}:{subscription_port}; listening on UDP {udp.getsockname()[1]}; Ctrl+C to stop.")
        else:
            print(f"Listening for broadcast telemetry on {args.bind}:{args.port}. Ctrl+C to stop.")

        deadline = time.monotonic() + args.duration if args.duration is not None else None
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if subscription and time.monotonic() >= next_renewal:
                    udp.sendto(b"POSE_SUBSCRIBE", subscription)
                    next_renewal = time.monotonic() + 1.0
                try:
                    data, sender = udp.recvfrom(65535)
                except socket.timeout:
                    continue
                received_at = time.time()
                try:
                    message = json.loads(data)
                    if not isinstance(message, dict):
                        raise ValueError("packet is not a JSON object")
                    formatted = format_packet(message, sender, received_at)
                    if not args.quiet:
                        print(formatted, flush=True)
                    if recording is not None:
                        json.dump({'received_at': received_at, 'sender': sender[0],
                                   'sender_port': sender[1], 'packet': message}, recording)
                        recording.write('\n')
                        recording.flush()
                    received += 1
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    invalid += 1
                    print(f"Invalid packet from {sender[0]}: {error}", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if subscription:
                try:
                    udp.sendto(b"POSE_UNSUBSCRIBE", subscription)
                except OSError:
                    pass

    print(f"\nReceived {received} valid packet(s); {invalid} invalid packet(s).")


if __name__ == "__main__":
    main()
