"""Run the shared live VSSS provider until interrupted."""
import argparse

from vsss._provider import Provider


class ForegroundProvider(Provider):
    def should_exit(self):
        return False


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        print('Live provider: press Ctrl+C to stop.', flush=True)
        ForegroundProvider().run()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        raise SystemExit(f'Cannot start live provider: {error}. '
                         'Another VSSS provider may already be active.') from error


if __name__ == '__main__':
    main()
