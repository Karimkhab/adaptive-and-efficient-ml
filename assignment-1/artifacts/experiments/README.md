# Controller check before the dynamic-obstacle fix

`controller_v1.json` and `controller_v1_map3.csv` preserve a real earlier run on
2026-10-25. The vehicle/environment already used the current cell spacing and
polyline observation correction. The driver used prediction horizon 1.3 s and
safety margin 0.25 for both static and moving obstacles.

The three original maps succeeded. On the additional crossing scenario (map 3)
the car stopped at x≈28.023, y=12, less than two radii from the obstacle's crossing
line x=30. At step 342 the moving obstacle touched the stopped vehicle; final
clearance was −0.000525. Reward was −82.11683.

The final controller retains those values for static objects and uses horizon
1.8 s / margin 0.75 for moving objects. Current results are in
`../evaluation.json`. The final scenario succeeds with positive clearance.
This is a regression fix on the same deterministic scenarios, not a statistical
estimate of general driving reliability.

The archived run used a goal acceptance radius of 1.2. The final package uses
radius 1 for the goal as well. This changes successful arrival times slightly;
the archived collision occurs far from the goal and is unaffected by that threshold.
