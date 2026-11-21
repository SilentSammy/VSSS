"""Minimal provider launch, observation and recording entry points."""
import argparse
import json
import logging
from threading import Event

from .models import encode
from .provider import VSSSProvider


def provider_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--endpoint", help="Provider discovery JSON path")
    parser.add_argument("--port", type=int, default=38475)
    return parser


def run_provider(provider, args):
    logging.basicConfig(level=logging.INFO)
    with provider:
        with provider.serve(endpoint=args.endpoint, port=args.port):
            print(f"{provider.mode} provider ready; Ctrl+C stops it", flush=True)
            try:
                # A bounded wait lets Windows deliver Python's pending SIGINT.
                while True:
                    Event().wait(.1)
            except KeyboardInterrupt:
                pass


def observe(argv=None):
    parser = argparse.ArgumentParser(description="Observe protocol-agnostic VSSS events")
    parser.add_argument("--endpoint")
    args = parser.parse_args(argv)
    with VSSSProvider.connect(args.endpoint) as client:
        subscription = client.subscribe(["gps", "commands", "lifecycle"],
                                        lambda event: print(json.dumps(encode(event)), flush=True))
        try:
            while not subscription.error:
                Event().wait(.2)
        except KeyboardInterrupt:
            pass
        if subscription.error:
            raise SystemExit(str(subscription.error))


def record(argv=None):
    from .recording import Recorder
    parser = argparse.ArgumentParser(description="Record domain events to new JSONL schema")
    parser.add_argument("path")
    parser.add_argument("--endpoint")
    args = parser.parse_args(argv)
    with VSSSProvider.connect(args.endpoint) as client, Recorder(client, args.path) as recorder:
        print("Recording; Ctrl+C completes the file", flush=True)
        try:
            while not recorder._subscription.error:
                Event().wait(.2)
        except KeyboardInterrupt:
            pass
