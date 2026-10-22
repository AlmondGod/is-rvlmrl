"""Render saved before/after programs with MuJoCo, physical state and exact code."""
import argparse
import bisect
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.cap_rl.audit import audit, equivalent
from experiments.cap_rl import task_mujoco as task


def font(size, mono=False):
    candidates = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                   "/System/Library/Fonts/Menlo.ttc"] if mono else
                  ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/System/Library/Fonts/Supplemental/Arial.ttf"])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render(directory, destination, index=0):
    directory, destination = Path(directory), Path(destination)
    audit(directory)
    report = json.loads((directory/'test_comparison.json').read_text())
    layout = task.datasets()['test'][index]
    rows = [report['before'][index], report['after'][index]]
    replays = [task.execute(layout, row['completion'], record_frames=True) for row in rows]
    for row, replay in zip(rows, replays):
        if not equivalent(row['final_state'], replay['final_state'], 1e-8):
            raise ValueError('Physical replay differs from saved episode')
    m = mujoco.MjModel.from_xml_string(task.XML)
    m.site_pos[m.site('station1').id, :2] = task.meters(layout.first)
    m.site_pos[m.site('station2').id, :2] = task.meters(layout.second)
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0]
    camera.distance = 1.55
    camera.azimuth = 90
    camera.elevation = -65
    heading, body, codefont = font(24), font(16), font(17, True)
    times = [[f['time'] for f in r['frames']] for r in replays]
    end = max(t[-1] for t in times)
    images = []
    # 40 Hz recorded physics, displayed at half speed (20 fps).
    sample_times = list(np.arange(0, end+.025, .025)) + [end]*40
    with mujoco.Renderer(m, height=360, width=480) as renderer:
        for elapsed in sample_times:
            canvas = Image.new('RGB', (960, 720), '#101820')
            draw = ImageDraw.Draw(canvas)
            for panel, (row, replay) in enumerate(zip(rows, replays)):
                offset = panel*480
                k = min(len(times[panel])-1, max(0, bisect.bisect_right(times[panel], elapsed)-1))
                state = replay['frames'][k]
                d.qpos[:] = state['position']
                d.qvel[:] = state['velocity']
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (offset, 65))
                ended = elapsed >= times[panel][-1]-1e-8
                status = ('SUCCESS' if row['success'] else 'FAIL: '+str(row['error'] or 'incomplete')) if ended else 'running'
                color = '#77dda2' if ended and row['success'] else '#ffbb83' if ended else '#dce8ef'
                draw.text((offset+18, 10), 'Before training' if panel == 0 else 'After training', font=heading, fill='#f4f7fa')
                draw.text((offset+18, 41), status, font=body, fill=color)
                draw.text((offset+18, 429), f"t={state['time']:.2f}s  speed={np.linalg.norm(state['velocity']):.3f} m/s  docked={state['docked']}/2", font=body, fill='#dce8ef')
                p = state['position']
                draw.text((offset+18, 452), f"position=({p[0]:+.3f}, {p[1]:+.3f}) m", font=body, fill='#dce8ef')
                lines = row['completion'].strip().splitlines()
                if lines and lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1] == '```':
                    lines = lines[:-1]
                call_line, active = -1, state['call_index']
                for n, line in enumerate(lines[:10]):
                    if line.strip() and not line.lstrip().startswith('#'):
                        call_line += 1
                    y = 484+n*20
                    if line.strip() and call_line == active:
                        draw.rectangle((offset+12, y-1, offset+468, y+19), fill='#274454')
                    draw.text((offset+18, y), line[:45], font=codefont, fill='#f4f7fa')
                if len(lines) > 10:
                    draw.text((offset+18, 684), '... full program in episode JSON', font=body, fill='#dce8ef')
            images.append(np.asarray(canvas))
    destination.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(destination.with_suffix('.mp4')), images, fps=20, codec='libx264', quality=8)
    # A static last-frame preview is unambiguous when a client cannot play video.
    Image.fromarray(images[-1]).save(destination.with_suffix('.png'))
    imageio.mimsave(str(destination.with_suffix('.gif')), images[::3], duration=150, loop=1)
    evidence = {'test_index': index, 'layout_id': layout.key,
                'layout_m': {k: task.meters(getattr(layout, k)) for k in ('start', 'first', 'second')},
                'before': rows[0], 'after': rows[1],
                'render_note': 'Actual MuJoCo renders; half-speed playback; terminated episodes freeze.'}
    destination.with_suffix('.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'index': index, 'layout_id': layout.key,
                      'before_success': rows[0]['success'], 'after_success': rows[1]['success'],
                      'video': str(destination.with_suffix('.mp4'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    render(args.directory, args.out, args.index)
