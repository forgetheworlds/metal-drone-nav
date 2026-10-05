"""Declared obstacle centers in metres at simulation time in seconds."""
import math


def position(kind, center, size, velocity, time):
    if kind == 3:
        speed = math.sqrt(sum(v * v for v in velocity))
        if speed <= 1e-8 or size[1] <= 1e-8:
            return tuple(center)
        offset = size[1] * math.sin(time * speed / size[1] + size[2]) / speed
        return tuple(center[j] + velocity[j] * offset for j in range(3))
    return tuple(center[j] + velocity[j] * time for j in range(3))
