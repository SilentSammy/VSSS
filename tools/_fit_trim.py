"""Fit a proposed trim matrix from measured signed single-axis pulse responses.

Input CSV columns: command_x, command_y, command_w, forward_m, left_m,
heading_deg, and optionally duration_s. Responses are net displacement after the car has stopped, rotated
into its starting robot frame. Input pulses must use the same known trim matrix.
This offline tool never connects to or moves the car.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from vsss._ble_client import normalize_trim_matrix


def inverse(matrix):
    work = [list(map(float, row)) + [float(i == j) for j in range(3)]
            for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda i: abs(work[i][col]))
        if abs(work[pivot][col]) < 1e-8:
            raise ValueError('response matrix is singular or too small')
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(3):
            if row != col:
                scale = work[row][col]
                work[row] = [value - scale * base
                             for value, base in zip(work[row], work[col])]
    return [row[3:] for row in work]


def fit(rows, current_trim):
    current = normalize_trim_matrix(current_trim)
    numerator = [[0.] * 3 for _ in range(3)]
    denominator = [0.] * 3
    signs = [set() for _ in range(3)]
    count = 0
    for row in rows:
        try:
            command = [float(row[key]) for key in ('command_x', 'command_y', 'command_w')]
            response = [float(row[key]) for key in ('forward_m', 'left_m', 'heading_deg')]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('each pulse needs numeric command and response columns') from error
        if not all(math.isfinite(value) for value in command + response):
            raise ValueError('pulse data must be finite')
        try:
            duration_value = row.get('duration_s', 1.)
            duration = float(1. if duration_value in (None, '') else duration_value)
        except (TypeError, ValueError) as error:
            raise ValueError('pulse duration must be positive and finite') from error
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('pulse duration must be positive and finite')
        response = [value / duration for value in response]
        active = [i for i, value in enumerate(command) if abs(value) > 1e-9]
        if len(active) != 1 or abs(command[active[0]]) > 1:
            raise ValueError('each pulse must command exactly one axis within [-1, 1]')
        axis = active[0]
        signs[axis].add(1 if command[axis] > 0 else -1)
        denominator[axis] += command[axis] ** 2
        for output in range(3):
            numerator[output][axis] += response[output] * command[axis]
        count += 1
    if any(axis_signs != {-1, 1} for axis_signs in signs):
        raise ValueError('both positive and negative pulses are required on every axis')
    response = [[numerator[out][axis] / denominator[axis] for axis in range(3)]
                for out in range(3)]
    response_inverse = inverse(response)
    # The observed response already includes the existing trim. Compose the
    # incremental correction with that trim to avoid discarding calibration.
    incremental = [[response_inverse[r][c] * response[c][c] for c in range(3)]
                   for r in range(3)]
    proposed = [[sum(current[r][k] * incremental[k][c] for k in range(3))
                 for c in range(3)] for r in range(3)]
    proposed = normalize_trim_matrix(proposed)
    return {'samples': count, 'response_matrix': response,
            'starting_trim': current, 'matrix': proposed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('--starting-trim', required=True, type=Path,
                        help='JSON matrix or object with matrix field, read from robot')
    parser.add_argument('--output', required=True, type=Path,
                        help='new JSON output file; never overwrites an existing file')
    args = parser.parse_args()
    trim = json.loads(args.starting_trim.read_text(encoding='utf-8'))
    if isinstance(trim, dict):
        trim = trim['matrix']
    with args.samples.open(newline='', encoding='utf-8') as source:
        result = fit(csv.DictReader(source), trim)
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(result, destination, indent=2)
        destination.write('\n')
    print('Proposal saved to', args.output, '(not applied to robot)')


if __name__ == '__main__':
    main()
