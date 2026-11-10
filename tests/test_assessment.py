import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests.experiments.assess import assess, response


def example():
    rows=[]
    def row(t,event,phase,angle,w):
        rows.append(dict(t_s=str(t),received_t_s=str(t),event=event,phase=phase,
                         trial='1',pose_valid='True',x_m='0',y_m='0',heading_deg=str(angle),
                         input_x='0',input_y='0',input_w=str(w)))
    row(0,'telemetry','pre_pulse',179,0)
    row(.1,'control','pulse',179,.5)
    row(.2,'telemetry','pulse',-179,.5)
    row(.3,'telemetry','pulse',-170,.5)
    row(.4,'telemetry','pulse',-165,.5)
    row(.5,'control','settle',-165,0)
    row(.5,'stop_request','settle',-165,0)
    row(.55,'stop_ack','settle',-165,0)
    for t in (.6,.7,.8,.95):
        row(t,'telemetry','settle',-161,0)
    return rows


class AssessmentTests(unittest.TestCase):
    def test_response_unwraps_angles_and_includes_post_stop_motion(self):
        result=response(example(),1,'synthetic')
        self.assertAlmostEqual(result['heading_change_deg'],20.)
        self.assertAlmostEqual(result['w_command_area'],.2)
        self.assertAlmostEqual(result['effective_yaw_gain_deg'],100.)
        self.assertAlmostEqual(result['onset_receipt_delay_s'],.2)
        self.assertAlmostEqual(result['stationary_receipt_delay_s'],.1)
        self.assertAlmostEqual(result['stop_ack_delay_s'],.05)

    def test_curation_excludes_rejected_trial_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            original=directory/'raw.csv'
            rows=example()
            rows.append(dict(rows[-1],trial='2',phase='aborted'))
            with original.open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            before=original.read_bytes()
            selection=directory/'selection.json'
            selection.write_text(json.dumps(dict(sources=[dict(csv='raw.csv',include_trials=[1])])))
            with contextlib.redirect_stdout(io.StringIO()):
                results=assess(selection)
            self.assertEqual(len(results),1)
            self.assertEqual(original.read_bytes(),before)
            with (directory/'accepted_events.csv').open() as file:
                accepted=list(csv.DictReader(file))
            self.assertTrue(all(row['trial']=='1' for row in accepted))
