"""Plot live or recorded VSSS field telemetry."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from vsss.provider import BagProvider, LiveProvider


MARKER_COLORS = (
    (173, 173, 255), (165, 214, 255), (182, 255, 253),
    (191, 255, 202), (255, 246, 155), (255, 196, 160), (255, 178, 189),
)


class FieldPlotter:
    """Small OpenCV field renderer adapted from TE2004B_AD26's plotter2d."""

    def __init__(self, field_width=1.6, field_height=3.0, width=720):
        if min(field_width, field_height, width) <= 0:
            raise ValueError('field dimensions and plot width must be positive')
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError('vsss-plot requires the plot extra: pip install .[plot]') from error
        self.cv2, self.np = cv2, np
        self.field_width, self.field_height = float(field_width), float(field_height)
        self.margin = 38
        self.width = int(width)
        self.scale = (self.width - 2 * self.margin) / self.field_width
        self.height = round(self.field_height * self.scale) + 2 * self.margin
        self.background = np.full((self.height, self.width, 3), (238, 238, 238), np.uint8)
        left, top = self.to_pixel(-self.field_width / 2, self.field_height / 2)
        right, bottom = self.to_pixel(self.field_width / 2, -self.field_height / 2)
        cv2.rectangle(self.background, (left, top), (right, bottom), (72, 112, 72), -1)
        cv2.rectangle(self.background, (left, top), (right, bottom), (245, 245, 245), 2)
        cv2.line(self.background, self.to_pixel(-self.field_width / 2, 0),
                 self.to_pixel(self.field_width / 2, 0), (210, 210, 210), 1)
        cv2.line(self.background, self.to_pixel(0, -self.field_height / 2),
                 self.to_pixel(0, self.field_height / 2), (210, 210, 210), 1)

    def to_pixel(self, x, y):
        return (round(self.width / 2 + float(x) * self.scale),
                round(self.height / 2 - float(y) * self.scale))

    def render(self, packet):
        cv2, np = self.cv2, self.np
        canvas = self.background.copy()
        marker_size = max(14, round(.09 * self.scale))
        for marker in packet.get('markers', ()):
            marker_id = int(marker['id'])
            center = np.asarray(self.to_pixel(marker['x'], marker['y']), dtype=float)
            angle = math.radians(float(marker['angle']))
            forward = np.array((math.cos(angle), -math.sin(angle)))
            right = np.array((-forward[1], forward[0]))
            half = marker_size / 2
            corners = np.asarray((center + forward * half + right * half,
                                  center + forward * half - right * half,
                                  center - forward * half - right * half,
                                  center - forward * half + right * half), dtype=np.int32)
            color = MARKER_COLORS[marker_id % len(MARKER_COLORS)]
            cv2.fillConvexPoly(canvas, corners, color)
            cv2.polylines(canvas, [corners], True, (35, 35, 35), 2, cv2.LINE_AA)
            tip = tuple(np.rint(center + forward * half).astype(int))
            cv2.line(canvas, tuple(center.astype(int)), tip, (25, 25, 25), 2, cv2.LINE_AA)
            label = str(marker_id)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
            cv2.putText(canvas, label, (round(center[0] - tw / 2), round(center[1] + th / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (10, 10, 10), 1, cv2.LINE_AA)
        for ball in packet.get('balls', ()):
            center = self.to_pixel(ball['x'], ball['y'])
            cv2.circle(canvas, center, max(5, round(.0215 * self.scale)), (40, 150, 255), -1,
                       cv2.LINE_AA)
            cv2.circle(canvas, center, max(5, round(.0215 * self.scale)), (20, 80, 150), 1,
                       cv2.LINE_AA)
        state = 'board detected' if packet.get('board_detected') else 'board not detected'
        cv2.putText(canvas, state, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (30, 30, 30), 1, cv2.LINE_AA)
        return canvas


def provider_from_args(args):
    if args.bag:
        return BagProvider(args.bag, topics=('udp',), speed=args.speed)
    vision = dict(server=None if args.broadcast else args.server, port=args.port,
                  listen_port=args.port if args.broadcast else 0,
                  subscription_port=args.subscription_port or args.port + 1)
    return LiveProvider(('udp',), vision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--bag', type=Path, help='replay a vsss-observer JSONL bag')
    source.add_argument('--broadcast', action='store_true', help='plot passive UDP broadcast')
    parser.add_argument('--speed', type=float, default=1.0, help='bag playback speed (default: 1)')
    parser.add_argument('--server', default='192.168.12.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--subscription-port', type=int)
    parser.add_argument('--field-width', type=float, default=1.6)
    parser.add_argument('--field-height', type=float, default=3.0)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--duration', type=float,
                        help='stop after this many seconds (useful for headless live plots)')
    parser.add_argument('--output', type=Path,
                        help='headless mode: save the final rendered frame instead of opening a window')
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    if args.output and args.output.exists():
        parser.error('--output must be a new file')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive and finite')
    plotter = FieldPlotter(args.field_width, args.field_height, args.width)
    cv2 = plotter.cv2
    provider = provider_from_args(args)
    latest = None
    window = 'VSSS field'
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    try:
        with provider:
            while not provider.finished and (deadline is None or time.monotonic() < deadline):
                event = provider.read(.02)
                if event and event.get('type') == 'udp' and event.get('valid'):
                    latest = plotter.render(event['packet'])
                if args.output is None and latest is not None:
                    cv2.imshow(window, latest)
                    key = cv2.waitKey(1) & 0xff
                    if key in (ord('q'), 27):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                elif args.output is None:
                    cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        if args.output is None:
            cv2.destroyAllWindows()
    if args.output:
        if latest is None:
            raise RuntimeError('provider produced no valid UDP frames')
        if not cv2.imwrite(str(args.output), latest):
            raise OSError(f'could not write plot image: {args.output}')
        print('Saved', args.output)
    elif args.bag:
        print(f'Replayed {args.bag} in {time.monotonic() - started:.2f}s')


if __name__ == '__main__':
    main()
