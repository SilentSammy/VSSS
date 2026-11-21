"""Two-frame scenario setup, reproducibility, and compatibility with bag seeding."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from vsss._sim import SimWorld, load_config
from vsss._sim_scenarios import apply_frames, goalie_frames, save_scenario


class ScenarioTests(unittest.TestCase):
    def test_example_starts_at_first_frame_with_derived_velocity(self):
        config = load_config(scenario='scenarios/goalie.jsonl')
        world = SimWorld(config)
        self.assertAlmostEqual(world.balls[0].x, -.25)
        self.assertAlmostEqual(world.balls[0].vx, .2)
        self.assertAlmostEqual(world.balls[0].vy, .8)
        self.assertEqual(world.cars[1].y, 1.2)

    def test_car_matching_uses_ids_and_angle_wraps_at_180(self):
        a = dict(time=0, cars=[dict(id=4, x=0, y=0, angle=179)], balls=[])
        b = dict(time=.1, cars=[dict(id=4, x=.01, y=-.02, angle=-179)], balls=[])
        config = load_config()
        apply_frames(config, a, b)
        world = SimWorld(config)
        self.assertAlmostEqual(world.cars[4].vx, .1)
        self.assertAlmostEqual(world.cars[4].vy, -.2)
        self.assertAlmostEqual(math.degrees(world.cars[4].omega), 20)

    def test_saved_resolved_random_start_round_trips_without_overwriting(self):
        config = load_config(scenario='goalie', seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shot.jsonl'
            save_scenario(config, path)
            restored = load_config(scenario=path)
            with self.assertRaises(FileExistsError):
                save_scenario(config, path)
        for key in ('field', 'random_seed', 'wait_for_control', 'cars', 'balls'):
            self.assertEqual(config[key], restored[key])

    def test_random_shots_aim_at_positive_y_edge_and_repeat_by_seed(self):
        self.assertEqual(goalie_frames(42), goalie_frames(42))
        self.assertNotEqual(goalie_frames(42), goalie_frames(43))
        for seed in range(100):
            world = SimWorld(load_config(scenario='goalie', seed=seed))
            ball = world.balls[0]
            self.assertLessEqual(ball.y, 0)
            self.assertGreater(ball.vy, 0)
            self.assertAlmostEqual(math.hypot(ball.vx, ball.vy), .8)
            end_x = ball.x + ball.vx / ball.vy * (world.height / 2 - ball.y)
            self.assertLessEqual(abs(end_x), .55 + 1e-9)

    def test_waiting_keeps_motion_frozen_but_feedback_fresh(self):
        world = SimWorld(load_config(scenario='goalie', seed=1))
        ball = world.balls[0]
        start = ball.x, ball.y, ball.vx, ball.vy
        for _ in range(50):
            world.step(.02)
        self.assertEqual((ball.x, ball.y, ball.vx, ball.vy), start)
        self.assertGreater(world.feedback_sequence, 0)
        world.waiting_for_control = False
        world.step(.02)
        self.assertGreater(ball.y, start[1])

    def test_bad_frame_pairs_are_rejected(self):
        rows = goalie_frames(1)
        a, b = rows[1:]
        cases = [dict(b, time=0), dict(b, time=float('nan')), dict(b, cars=[]), dict(b, balls=[]),
                 dict(b, cars=[b['cars'][0]] * 2)]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_frames(load_config(), a, bad)

    def test_bag_motion_is_opt_in_and_ordinary_bags_still_use_one_frame(self):
        rows = [dict(type='bag_header', provider={})]
        for t, y in ((0, -.4), (.1, -.32)):
            rows.append(dict(type='udp', valid=True, packet=dict(board_detected=True, timestamp=t,
                markers=[dict(id=1, x=0, y=1.2, angle=0)], balls=[dict(x=0, y=y)])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shot.bag'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            stationary = load_config(bag=path)
            moving = load_config(scenario=path)
        self.assertNotIn('vy', stationary['balls'][0])
        self.assertAlmostEqual(moving['balls'][0]['vy'], .8)
        self.assertEqual(moving['balls'][0]['y'], -.4)

    def test_incomplete_file_and_conflicting_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            path.write_text(json.dumps(goalie_frames(1)[0]))
            with self.assertRaises(ValueError):
                load_config(scenario=path)
        with self.assertRaises(ValueError):
            load_config(bag='unused', scenario='goalie')
        with self.assertRaises(ValueError):
            load_config(scenario='goalie', seed=-1)

    def test_cli_saved_scenario_waits_through_observation_and_car_connection(self):
        from vsss import Car, gps
        from vsss.sim import SimClient
        with tempfile.TemporaryDirectory() as directory, socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
            probe.close()
            saved = Path(directory) / 'saved.jsonl'
            token = Path(tempfile.gettempdir()) / f'vsss-provider-{port}.token'
            with patch.dict(os.environ, VSSS_PROVIDER_PORT=str(port)):
                process = subprocess.Popen([sys.executable, '-m', 'tools.vsss_sim', '--scenario', 'goalie',
                    '--seed', '42', '--save-scenario', str(saved)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not token.exists() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail('Scenario CLI failed: ' + ''.join(process.communicate()))
                        time.sleep(.01)
                    self.assertTrue(token.exists())
                    with SimClient() as editor, Car(1) as car:
                        start = editor.state()['balls'][0]
                        time.sleep(.2)
                        self.assertEqual(editor.state()['balls'][0], start)
                        car.set_velocity(x=0., y=0., w=0.)
                        time.sleep(.3)
                        self.assertGreater(editor.state()['balls'][0]['y'], start['y'] + .05)
                    restored = load_config(scenario=saved)
                    self.assertEqual(restored['random_seed'], 42)
                    self.assertTrue(restored['wait_for_control'])
                finally:
                    Car.close_all()
                    gps.close()
                    process.terminate()
                    process.communicate(timeout=5)
                    token.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
