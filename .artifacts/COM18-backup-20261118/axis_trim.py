"""Linear axis gains and cross-axis compensation."""
import struct


class AxisTrim:
    def __init__(self, matrix=None, on_change=None):
        self.matrix = tuple(matrix or (1., 0., 0., 0., 1., 0., 0., 0., 1.))
        self.on_change = on_change

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
        matrix = self.matrix
        return tuple(max(-1., min(1., sum(matrix[3*i+j] * velocity[j] for j in range(3))))
                     for i in range(3))
