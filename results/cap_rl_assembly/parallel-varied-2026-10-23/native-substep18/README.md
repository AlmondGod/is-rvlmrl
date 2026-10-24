# Experimental fused native physics loop

The prototype moves the existing 40 internal physics steps per control update
into one C call, including finite-strength nail forces, MuJoCo integration,
penetration checks, actuator saturation and the signed nail-support observer.
The timestep, controls, contact model, force law and strict reward are unchanged.
This remains CPU physics with GPU camera rendering.

A full Linux success replay matched all 6,482 by 155 recorded values exactly,
along with every non-timing result. Wall time was 780.51 seconds versus the
previous vectorized run's 1,143.33 seconds, an observed 1.465× ratio on a shared
host. This is one case, not isolated throughput or a whole-RL-loop benchmark.
The comparison includes the previously verified native pixel and connector
optimizations too.

The synthetic numerical check covered 100 control steps, all compared dynamic
arrays, three irreversible nail failures, support measurements and joint-command
overrides. It matched exactly and measured about 1.97× faster environment steps.
The compiled connector arithmetic also passed its 512-case compatibility check.

The first full-run prototype failed immediately because its adapter omitted the
intermediate joint-command forwarding method. That failed attempt is preserved
in the evidence archive; its short runtime is not a speed result. The corrected
adapter includes that method and the regression fixture exercises it.

No active training run uses this prototype. A full failed-episode comparison,
production integration and isolated throughput measurements remain outstanding.
