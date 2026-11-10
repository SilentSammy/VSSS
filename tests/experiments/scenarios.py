"""Scenarios share recording, bounds, recentering, and controller handling."""
import json

from .correction_fit import fit_correction
from .session import Experiment


class CorrectionExperiment(Experiment):
    def run(self, plan):
        self.group_size = len(plan)//2
        self.active_group = None
        self.candidate_matrix = self.args.trim_matrix
        return super().run(plan)

    def before_trial(self, trial):
        group = 'identity' if trial <= self.group_size else 'candidate'
        if group == self.active_group:
            return False
        self.stop()
        if group == 'candidate' and self.args.auto_fit:
            self.log.flush()
            fit = fit_correction(self.directory/'samples.csv', self.group_size)
            self.candidate_matrix = fit['matrix']
            (self.directory/'fitted_correction.json').write_text(json.dumps(fit, indent=2))
            print('  Fitted correction matrix:', flush=True)
            for row in self.candidate_matrix:
                print('   ', ' '.join(f'{value:+.4f}' for value in row), flush=True)
        matrix = None if group == 'identity' else self.candidate_matrix
        self.car.set_trim(matrix)
        self.log.record('trim_configuration', note=group)
        self.active_group = group
        return True


class WaypointExperiment(Experiment):
    def run(self, plan):
        with self.recorded_run():
            self.rest(self.args.settle, 'baseline', 0)
            self.recenter(0)
            for trial, waypoint in enumerate(plan, 1):
                if trial < self.args.start_trial:
                    continue
                target = tuple(self.home_target[i] if value is None else value
                               for i,value in enumerate(waypoint.target))
                self.log.set_segment(waypoint.name, 1)
                self.move_to(target, trial)
                self.rest(self.args.settle, 'waypoint_rest', trial)
                if trial % self.args.recenter_every == 0:
                    self.recenter(trial)
            self.recenter(len(plan)+1)
            self.rest(self.args.settle, 'final_rest', len(plan)+1)


SCENARIOS = dict(strides=Experiment, correction=CorrectionExperiment, waypoints=WaypointExperiment)
