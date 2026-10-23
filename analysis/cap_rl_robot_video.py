"""Render saved before/after programs with MuJoCo, physical state and exact code."""
import argparse
import ast
import bisect
import json
from pathlib import Path
import textwrap

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.cap_rl.audit import audit, equivalent
from experiments.cap_rl.tasks import get_task
from experiments.cap_rl.robot_lift import model, Robot


def font(size, mono=False):
    candidates = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                   "/System/Library/Fonts/Menlo.ttc"] if mono else
                  ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/System/Library/Fonts/Supplemental/Arial.ttf"])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(directory, destination, index=0, validation_step=None):
    directory, destination = Path(directory), Path(destination)
    metadata = json.loads((directory/"metadata.json").read_text())
    task = get_task(metadata.get("task", "robot_lift"))
    placement = metadata.get("task") == "frame_place"
    if validation_step is None:
        report = json.loads((directory/'test_comparison.json').read_text())
        layout = task.datasets()['test'][index]
        rows = [report['before'][index], report['after'][index]]
    else:
        layout = task.datasets()['validation'][index]
        rows = [json.loads((directory/name).read_text())['rows'][index] for name in
                ('validation_baseline.json', f'validation_{validation_step:04d}.json')]
    if any(row['layout_id'] != layout.key for row in rows):
        raise ValueError('Layout mismatch')
    replays = [task.execute(layout, row['completion'], record_frames=True) for row in rows]
    for row, replay in zip(rows, replays):
        if row['success'] != replay['success'] or abs(row['reward']-replay['reward']) > 1e-6:
            raise ValueError('Physical replay differs from saved episode')
        if 'frames' not in replay:
            replay['frames'] = Robot(*layout.center, start=layout.start, record=True).frames
    m = model()
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.45, -.20, .20]
    camera.distance = 1.4
    camera.azimuth = 145
    camera.elevation = -25
    heading, body, codefont = font(24), font(16), font(17, True)
    times = [[f['time'] for f in r['frames']] for r in replays]
    end = max(t[-1] for t in times)
    images = []
    # Actual recorded 10 Hz robot states, played at real time.
    sample_times = list(np.arange(0, end+.1, .1)) + [end]*20
    with mujoco.Renderer(m, height=360, width=480) as renderer:
        for elapsed in sample_times:
            canvas = Image.new('RGB', (960, 768), '#101820')
            draw = ImageDraw.Draw(canvas)
            for panel, (row, replay) in enumerate(zip(rows, replays)):
                offset = panel*480
                k = min(len(times[panel])-1, max(0, bisect.bisect_right(times[panel], elapsed)-1))
                state = replay['frames'][k]
                d.qpos[:] = state['qpos']
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=camera)
                if placement:
                    # Rendering-only target outline: never added to collision physics.
                    goal = task.ORIGIN + np.array(layout.target)*.01
                    for delta, size in [((-.020, 0), (.001, .458, .001)),
                                        ((.020, 0), (.001, .458, .001)),
                                        ((0, -.458), (.020, .001, .001)),
                                        ((0, .458), (.020, .001, .001))]:
                        scene = renderer.scene
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                            np.array(size), np.r_[goal+delta, .001], np.eye(3).ravel(),
                            np.array([.2, .9, .9, 1.], dtype=np.float32))
                        scene.ngeom += 1
                canvas.paste(Image.fromarray(renderer.render()), (offset, 65))
                ended = elapsed >= times[panel][-1]-1e-8
                status = ('SUCCESS' if row['success'] else 'FAIL: '+str(row['error'] or ('placement criteria' if placement else 'lift criteria'))) if ended else 'running'
                color = '#77dda2' if ended and row['success'] else '#ffbb83' if ended else '#dce8ef'
                draw.text((offset+18, 10), 'Before training' if panel == 0 else 'After training', font=heading, fill='#f4f7fa')
                draw.text((offset+18, 41), status, font=body, fill=color)
                draw.text((offset+18, 429), f"t={state['time']:.1f}s  rail lift={(state['wood'][2]-.032)*100:.1f} cm", font=body, fill='#dce8ef')
                p = state['wood']
                detail = f"finger contact={state['grip_force']:.1f} N"
                if placement:
                    target = task.ORIGIN + np.array(layout.target)*.01
                    detail += f"; target error={np.linalg.norm(np.array(p[:2])-target)*1000:.1f} mm"
                draw.text((offset+18, 452), detail, font=body, fill='#dce8ef')
                lines = row['completion'].strip().splitlines()
                if lines and lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1] == '```':
                    lines = lines[:-1]
                try:
                    task.parse_program(row['completion'])
                    call_lines = [node.lineno-1 for node in ast.parse('\n'.join(lines)).body]
                except (SyntaxError, ValueError, TypeError, RecursionError):
                    call_lines = []
                active = state['call_index']
                active_line = call_lines[active] if 0 <= active < len(call_lines) else -1
                wrapped = [(n, chunk) for n, line in enumerate(lines) for chunk in (textwrap.wrap(line, width=44) or [''])]
                for n, (source_line, line) in enumerate(wrapped[:10]):
                    y = 484+n*20
                    if source_line == active_line:
                        draw.rectangle((offset+12, y-1, offset+468, y+19), fill='#274454')
                    draw.text((offset+18, y), line, font=codefont, fill='#f4f7fa')
                if len(wrapped) > 10:
                    draw.text((offset+18, 684), '... full program in episode JSON', font=body, fill='#dce8ef')
                if placement:
                    draw.text((offset+18, 692), 'Cyan outline: target footprint', font=body, fill='#77dddd')
                draw.text((offset+18, 716), f'Gripper start: {layout.start} cm', font=body, fill='#77dda2')
                footer = (f'Rail {layout.center} → {layout.target}; lift {layout.lift_cm} cm' if placement else
                          f'Rail center: {layout.center} cm; lift {layout.lift_cm} cm')
                draw.text((offset+18, 739), footer, font=body, fill='#ffbb83')
            images.append(np.asarray(canvas))
    destination.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(destination.with_suffix('.mp4')), images, fps=10, codec='libx264', quality=8)
    # A static last-frame preview is unambiguous when a client cannot play video.
    Image.fromarray(images[-1]).save(destination.with_suffix('.png'))
    imageio.mimsave(str(destination.with_suffix('.gif')), images[::2], duration=200, loop=1)
    evidence = {'split': 'test' if validation_step is None else 'validation', 'index': index, 'layout_id': layout.key,
                'layout': task.asdict(layout),
                'before': rows[0], 'after': rows[1],
                'render_note': 'Actual MuJoCo renders; real-time playback; terminated episodes freeze.'}
    destination.with_suffix('.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'index': index, 'layout_id': layout.key,
                      'before_success': rows[0]['success'], 'after_success': rows[1]['success'],
                      'video': str(destination.with_suffix('.mp4'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--validation-step', type=int)
    args = parser.parse_args()
    render(args.directory, args.out, args.index, args.validation_step)
