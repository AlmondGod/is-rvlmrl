# Third-party assets and provenance

Only the two robot asset trees required by the frozen torso benchmark are
vendored. Original license files and attribution are retained.

| Asset tree | Source checkout | License |
|---|---|---|
| `third_party/mujoco_menagerie/universal_robots_ur10e` | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/universal_robots_ur10e) | [BSD-3-Clause](third_party/mujoco_menagerie/universal_robots_ur10e/LICENSE), ROS Industrial Consortium |
| `third_party/3-PoinTr/simulation_assets/ufactory_xarm7` | [3-PoinTr](https://github.com/adamhung60/3-PoinTr/tree/0fd50a4b9da4c89e41f3c581ecbf8e232c4d5b19/simulation_assets/ufactory_xarm7) | [BSD-3-Clause](third_party/3-PoinTr/simulation_assets/ufactory_xarm7/LICENSE), UFACTORY Inc. |

The recorded commits identify the local source checkouts. The file hashes in
`provenance/frozen_v9/manifest.json` identify the actual vendored bytes and are
authoritative if they differ from an upstream checkout. These simplified robot
descriptions are not vendor validation of this multi-arm embodiment.

MuJoCo, NumPy, SciPy, OpenCV, ImageIO, FFmpeg, vLLM and model weights are separate
dependencies governed by their own licenses. They are not relicensed here.
Qwen weights are not bundled; model revision and checksums are recorded in
`provenance/qwen_runtime.json`. No Codex executable, login credential, API key or
SSH key is distributed.

This snapshot does not assign a new license to first-party project code.

