"""Print or save controller parameters inferred from a recorded pulse CSV."""
import argparse
from datetime import date
import json
from pathlib import Path
import time

from tests.experiments.recommend_controller import recommend


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples',type=Path,nargs='?',help='existing samples.csv')
    parser.add_argument('--run',action='store_true',help='physically run the guarded calibration pulses')
    parser.add_argument('--car',type=int,default=1)
    parser.add_argument('--save',action='store_true',help='make this car use the recommendation')
    args=parser.parse_args(argv)
    if args.run and args.samples is not None:
        parser.error('provide an existing CSV or --run, not both')
    if not args.run and args.samples is None:
        parser.error('provide samples.csv or use --run')
    samples=args.samples
    if args.run:
        from tests.experiments.run import main as run_experiment
        directory=Path('tests/experiments/runs')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+f'_controller_cal_car{args.car}')
        status=run_experiment([
            '--execute','--experiment','strides','--profile','quick',
            '--axes','x','w','--levels','.35','--repeats','2',
            '--duration','.7','--settle','1','--telemetry','vsss',
            '--bounds-x','.8','--bounds-y','1.5','--center-x','0','--center-y','.75',
            '--heading','90','--recenter-every','1','--center-timeout','30',
            '--connect-timeout','45','--max-seconds','240','--car',str(args.car),
            '--output',str(directory)])
        if status:
            raise SystemExit(status)
        samples=directory/'samples.csv'
    result=recommend(samples)
    print(json.dumps(result,indent=2))
    if args.save:
        recommended=result['recommended']
        configuration=dict(
            controller='predictive',
            parameters=recommended,
            selected_at=date.today().isoformat(),
            source=str(samples),
            measured=result['measured'])
        path=Path(__file__).with_name('calibration')/f'controller_car{args.car}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(configuration,indent=2)+'\n',encoding='utf-8')
        print(f'Saved: {path.resolve()}')


if __name__=='__main__':main()
