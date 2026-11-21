"""Fit residual commands for each pair of axes after linear trim validation."""
import math

from mp.axis_trim import mixed_features
from tools._fit_trim import inverse

PAIRS = ((0, 1, 2), (0, 2, 1), (1, 2, 0))


def fit_mixed(rows, response, deadbands=None, step=1.):
    """Fit only the unwanted third-axis response, preserving active outputs.

    ``response`` is the measured 3x3 rate response AFTER linear trim.
    Both signs in all four quadrants and multiple levels are required.
    Coefficients map requested mixed features to compensating input commands.
    """
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError('mixed step must be in (0,1]')
    inv = inverse(response)
    coefficients = [[0.] * 6 for _ in range(3)]
    diagnostics = []
    for pair_index, (a, b, third) in enumerate(PAIRS):
        selected = [r for r in rows if r['pair'] == pair_index]
        signs = {(1 if r['command'][a] > 0 else -1,
                  1 if r['command'][b] > 0 else -1) for r in selected}
        levels = {round(abs(r['command'][a]), 4) for r in selected}
        if signs != {(-1, -1), (-1, 1), (1, -1), (1, 1)} or len(levels) < 2:
            raise ValueError('each pair needs all sign quadrants at two levels')
        if deadbands is not None and all(abs(r['response'][third])<=deadbands[third]
                                         for r in selected):
            diagnostics.append(dict(pair=pair_index, skipped='baseline meets drift tolerance'))
            continue
        g00 = g01 = g11 = z0 = z1 = 0.
        for row in selected:
            duration = row['duration_s']
            if duration <= 0 or row['frames'] < 3:
                raise ValueError('insufficient pulse duration or fresh frames')
            feature = mixed_features(row['command'])[2*pair_index:2*pair_index+2]
            f0, f1 = feature
            # Isolate the residual the user wants zero for the uncommanded axis.
            unwanted_rate = row['response'][third] / duration
            g00 += f0*f0
            g01 += f0*f1
            g11 += f1*f1
            z0 += f0*unwanted_rate
            z1 += f1*unwanted_rate
        determinant = g00*g11-g01*g01
        if determinant < 1e-8:
            raise ValueError('mixed sample design is rank deficient')
        beta = ((z0*g11-z1*g01)/determinant,
                (z1*g00-z0*g01)/determinant)
        for output in range(3):
            for j in range(2):
                coefficients[output][2*pair_index+j] = -inv[output][third]*beta[j]
        diagnostics.append(dict(pair=pair_index, unwanted_rate_coefficients=beta))
    undamped = [round(v, 6) for row in coefficients for v in row]
    flat = [round(v*step, 6) for row in coefficients for v in row]
    if not all(math.isfinite(v) and abs(v) <= 1 for v in flat):
        raise ValueError('mixed proposal exceeds coefficient bounds')
    # Never fit a proposal that would operate in the firmware correction clamp
    # anywhere inside the measured box (including triples of axes).
    max_levels = [max(abs(r['command'][i]) for r in rows) for i in range(3)]
    import itertools
    for command in itertools.product(*[(-v, 0., v) for v in max_levels]):
        feature = mixed_features(command)
        if any(abs(sum(flat[6*i+j]*feature[j] for j in range(6))) > .15 for i in range(3)):
            raise ValueError('mixed proposal would exceed the correction clamp')
    result = dict(mixed_trim=flat, diagnostics=diagnostics,
                  calibrated_max_levels=max_levels)
    if step < 1:
        result.update(undamped_mixed_trim=undamped, step=step)
    return result
