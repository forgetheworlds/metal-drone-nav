"""Run the bank's smooth bounded sphere paths, independently of navigation."""
import json
from controller import Supervisor

# The isolated project receives the same small motion module next to this file.
from obstacle_motion import position


def main():
    supervisor = Supervisor()
    schedules = json.loads(supervisor.getCustomData())
    targets = []
    for schedule in schedules:
        node = supervisor.getFromDef(schedule["def"])
        if node is None:
            raise RuntimeError("missing mover " + schedule["def"])
        targets.append((schedule, node, node.getField("translation")))
    next_trace = 0.0
    while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
        time = supervisor.getTime()
        for schedule, node, field in targets:
            expected = position(schedule["kind"], schedule["center"], schedule["size"], schedule["velocity"], time)
            field.setSFVec3f(list(expected))
        if time + 1e-9 >= next_trace:
            for schedule, node, field in targets:
                xyz = node.getPosition()
                print(f"BOUNDED_MOVER time_s={time:.6f} def={schedule['def']} actual={xyz[0]:.9g},{xyz[1]:.9g},{xyz[2]:.9g}", flush=True)
            next_trace += .05


if __name__ == "__main__":
    main()
