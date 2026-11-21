"""Run a simulated VSSS fleet; use the usual car, GPS, teleop and observation tools."""
import argparse
from pathlib import Path

from vsss._sim import load_config
from vsss._sim_provider import SimProvider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', nargs='?', type=Path,
                        help='initialize from the bag header and first valid detected field frame')
    parser.add_argument('--config', type=Path, help='JSON world settings overriding the bundled defaults')
    parser.add_argument('--scenario', help='goalie, a scenario JSONL, or a bag using its first two field frames')
    parser.add_argument('--save-scenario', type=Path, help='save the resolved two-frame start to a new JSONL file')
    parser.add_argument('--ball-model', type=Path, help='ball model JSON or tools.ball_dynamics fit report')
    parser.add_argument('--seed', type=int, help='repeatable scenario and ball wobble seed (default: random)')
    parser.add_argument('--cars', type=int, nargs='+', metavar='ID',
                        help='override the fleet with these IDs, placed in a row facing +Y')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.ball_model, bag=args.bag, scenario=args.scenario, seed=args.seed)
        if args.cars is not None:
            count = len(args.cars)
            config['cars'] = [dict(id=number, x=(index - (count - 1) / 2) * .22,
                                   y=-.45, angle=90) for index, number in enumerate(args.cars)]
        provider = SimProvider(config)
        if args.save_scenario is not None:
            from vsss._sim_scenarios import save_scenario
            save_scenario(provider.world.config, args.save_scenario)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f'Simulation provider: cars {provider.available_cars()}; seed {provider.world.seed}; press Ctrl+C to stop.', flush=True)
    if provider.world.waiting_for_control:
        print('Scenario is waiting for the first car velocity/waypoint command.', flush=True)
    try:
        provider.run()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        parser.exit(1, f'Cannot start simulation provider: {error}. '
                       'Another VSSS provider may already be active.\n')


if __name__ == '__main__':
    main()
