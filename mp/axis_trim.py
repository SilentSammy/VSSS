"""Linear trim plus mixed-axis residual compensation."""
import struct


def mixed_features(velocity):
    """Odd quadratic terms; every term vanishes on single-axis commands.

    Order: x|y|, y|x|, x|w|, w|x|, y|w|, w|y|.
    Signed terms distinguish opposite diagonals unlike a plain x*y term.
    """
    x, y, w = velocity
    return (x * abs(y), y * abs(x), x * abs(w), w * abs(x),
            y * abs(w), w * abs(y))


class AxisTrim:
    def __init__(self, matrix=None, on_change=None, mixed=None):
        self.matrix = tuple(matrix or (1., 0., 0., 0., 1., 0., 0., 0., 1.))
        self.on_change = on_change
        self.mixed = tuple(mixed or (0.,) * 18)

    def read(self):
        return struct.pack('<9f', *self.matrix)

    def write(self, data):
        if len(data) == 36:
            matrix = struct.unpack('<9f', data)
            if all((0 if i in (0, 4, 8) else -2) <= value <= 2
                   for i, value in enumerate(matrix)):
                if matrix != self.matrix:
                    self.matrix = matrix
                    if self.on_change is not None:
                        self.on_change(matrix)
        return self.read()  # Restore the canonical value even for invalid writes.

    def apply(self, velocity):
        velocity = tuple(velocity)
        features = mixed_features(velocity)
        # Corrections operate in the requested-axis frame, before linear trim.
        # Bound the extra command to 0.15 per axis; the linear fit is retained.
        velocity = tuple(value + max(-.15, min(.15, sum(
            self.mixed[6*i+j] * features[j] for j in range(6))))
                         for i, value in enumerate(velocity))
        matrix = self.matrix
        return tuple(max(-1., min(1., sum(matrix[3*i+j] * velocity[j] for j in range(3))))
                     for i in range(3))
