"""Fit a 3x3 feed-forward decoupling matrix from signed single-axis pulses."""
import math

from mecanum_client import normalize_trim_matrix
from .summarize import summarize


def _inverse(matrix):
    work = [list(row) + [float(i == j) for j in range(3)]
            for i, row in enumerate(matrix)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-9:
            raise ValueError('response matrix is singular; calibration motion was insufficient')
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(3):
            if row == column:
                continue
            scale = work[row][column]
            work[row] = [value-scale*base for value,base in zip(work[row],work[column])]
    return [row[3:] for row in work]


def fit_correction(samples_path, identity_trials):
    """Return M such that measured_response @ M retains gains but removes coupling."""
    rows = [row for row in summarize(samples_path)
            if row['trial'] <= identity_trials and row['pulse_finished']]
    if len(rows) < 6:
        raise ValueError(f'need six complete signed pulses; found {len(rows)}')
    numerators = [[0.0]*3 for _ in range(3)]
    denominators = [0.0]*3
    for row in rows:
        command = (row['input_x'], row['input_y'], row['input_w'])
        response = (row['net_forward_m'], row['net_left_m'], row['net_heading_deg'])
        active = [i for i,value in enumerate(command) if abs(value) > 1e-9]
        if len(active) != 1 or not all(math.isfinite(value) for value in response):
            continue
        axis = active[0]
        denominators[axis] += command[axis]**2
        for output in range(3):
            numerators[output][axis] += response[output]*command[axis]
    if any(value <= 0 for value in denominators):
        raise ValueError('calibration needs positive and negative pulses on all three axes')
    response = [[numerators[out][axis]/denominators[axis] for axis in range(3)]
                for out in range(3)]
    inverse = _inverse(response)
    matrix = [[inverse[row][column]*response[column][column] for column in range(3)]
              for row in range(3)]
    matrix = normalize_trim_matrix(matrix)
    return dict(
        method='least-squares signed single-axis displacement; preserves diagonal response',
        samples=len(rows),
        response_matrix=response,
        matrix=[list(row) for row in matrix],
    )
