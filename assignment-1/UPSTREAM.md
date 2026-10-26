# Sources and attribution

Starter repository: [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment),
commit `1056d711e166747cc485725e31f2535967b030dd`, inspected on 2026-10-25.

The original `src/env/base.py`, `src/env/tasks.py`, `src/utils/renderer.py`,
`src/utils/utils.py`, package initializers, `instructions/` and
`src/algorithms/readme.md` are retained unchanged. The original root readme is
retained as `UPSTREAM_README.md`.

New files implement the vehicle, demonstration controller, adapted renderer,
evaluation, video generation and tests. `src/main.py` is replaced with a CLI;
`src/utils/autoreset.py` stores its array-valued initial state as a dynamic PyTree
field instead of a static field. Dependencies are pinned for this implementation.

Course requirements: **Assignments.pdf**, posted 2026-09-30 in the course chat:
[source message](https://t.me/c/3789089672/80) (course-group access required).
Stage 1 also appears in the retained `instructions/stage_1.md`.

Kinematic bicycle equations were checked against the author's
[Robotics Toolbox documentation](https://petercorke.github.io/robotics-toolbox-python/mobile-vehicle-bicycle.html).
This implementation adds longitudinal acceleration/resistance and uses midpoint
position/heading integration; it does not import Robotics Toolbox code.

CPU installation reference: [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The exact environment used here is recorded separately in `requirements-lock.txt`
and `artifacts/evaluation.json`.

This package covers Assignment 1. There is no RL training or model checkpoint.
