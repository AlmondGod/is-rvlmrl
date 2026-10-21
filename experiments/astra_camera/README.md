# Astra camera/proprioception controller — cloud experiment

Status: code preparation only. **No camera-control performance result exists.**
The previously supplied RunPod endpoint rejected the configured SSH key. Cloud
deployment and runtime checks require a working authorized endpoint. No local
simulations, rendering, policy training or evaluation were run for this version.

This version follows the previous experiment's interpretation: Astra authors
the perception and feedback-control program. It is not an Astra API call on
each frame. The user has also been asked whether online model decisions were
intended instead; that would be a distinct controller using the same interface.

## Observation boundary

The controller runs in a separate spawned process. It receives only:

- Two fixed calibrated RGB cameras, 640×480, refreshed at 10 simulated Hz.
- Robot wrist pose, finger positions and velocities, and previous motor targets.
- Camera calibration, robot joint/actuator specifications, and simulation time.

It receives no object pose or velocity, contact force, contact normal, contact
Jacobian, depth image, renderer segmentation, object ID, randomization seed,
reward, success flag, or environment object. Wrist pose is treated as robot
proprioception, analogous to forward kinematics on an arm; the ideal wrist
carrier remains a limitation of this simulation.

Static task priors are explicit: known rail size, approximate workspace region,
wood appearance, tabletop height, and the same previously optimized grasp
offset/orientation and open/closed finger templates. These are not per-episode
ground-truth pose inputs. The controller is therefore a narrow known-object
vision-control pilot, not a general object-recognition model.

`worker.py` defines the exact observation allowlist. `harness.py` constructs
packets directly from cameras and robot-only state, rather than filtering a
large privileged observation. Hidden state is used separately by the evaluator
for physical scoring. This process separation is not an adversarial sandbox;
controller source must also be audited for forbidden dependencies/file reads.

## Fixed simulation and evaluation

Keep the previous accepted `firm_v1` contact parameters, 1 ms integration,
50 simulated Hz control, physical action limits and strict success criteria.
Only cameras are added to the environment. These contact parameters are still
uncalibrated against hardware; the original nominal soft-contact failure is not
erased. The camera coordinate conversion follows the official
[MuJoCo visualization documentation](https://mujoco.readthedocs.io/en/latest/programming/visualization.html).
Calibration projections and EGL rendering must be checked on the cloud host
before interpreting controller failures.

Development uses seeds 0–19. Iterate only on development episodes. Freeze and
hash the controller/perception sources before validation on seeds 10000–10019.
Only after a successful validation gate (at least 18/20 strict successes), run
held-out seeds 900000–900049. These match the previous teacher's seed sets for
paired comparison; they must not be used for controller development.

Run black-image and frozen-first-image ablations on development seeds. A black
image should trigger a safe hold/failure rather than exploiting a nominal object
location. A frozen-image result distinguishes initial visual localization from
continued visual feedback; it does not alone prove all feedback is visual,
because robot proprioception is intentionally retained.

Every episode records actual camera views, motor commands, robot feedback,
controller status, evaluator-only diagnostics, action timing, and errors.
Simulated control frequency is not a claim of real-time execution; wall-clock
command latency is measured separately.

## Cloud launch

`prepare_bundle.py` creates a small whitelisted source/asset archive without
running simulation or including SSH keys or unrelated data. On an authorized
cloud host, unpack it into a new experiment directory, then run:

```bash
PHYLA_CLOUD_RUN=1 bash experiments/astra_camera/cloud_run.sh --seeds 0 --steps 10 --out outputs/frame_assembly/astra_camera/cloud_smoke
PHYLA_CLOUD_RUN=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-camera/bin/python -m experiments.astra_camera.harness --seeds 0 1 2 3 4
```

The first command installs isolated dependencies and checks startup/initial
camera commands. Linux EGL rendering support must be available. The harness
refuses to execute simulations on macOS or without `PHYLA_CLOUD_RUN=1`.
Output folders refuse silent overwrites. Copy results off ephemeral cloud disk
before releasing the machine. Do not modify unrelated jobs or rent a new
machine without an appropriate user request.
