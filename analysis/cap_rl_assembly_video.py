"""Render recorded full-assembly states; no dynamics or policy inference."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def load(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    manifest = json.loads((directory / 'provenance.json').read_text())
    xml = ET.fromstring((directory / 'scene.xml').read_text())
    for element in xml.iter():
        filename = element.get('file')
        if filename:
            relative = 'third_party/' + filename.split('/third_party/', 1)[1]
            local = ROOT / relative
            if hashlib.sha256(local.read_bytes()).hexdigest() != manifest['sha256'][relative]:
                raise RuntimeError('Robot asset differs from recorded episode: ' + relative)
            element.set('file', str(local))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    # Display-only lighting. Recorded policy camera pixels and physics are not
    # modified; this replay uses the overview camera for human inspection.
    model.vis.headlight.ambient[:] = [.4, .4, .4]
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    recording = np.load(directory / 'states.npz')
    rows = recording['rows']
    if rows.ndim != 2 or not len(rows):
        raise ValueError('No recorded states to render')
    renderer = mujoco.Renderer(model, 480, 640)
    return dict(result=result, rows=rows, model=model, data=mujoco.MjData(model), renderer=renderer)


def panel(episode, seconds, title):
    rows, model, data = episode['rows'], episode['model'], episode['data']
    i = min(max(int(np.searchsorted(rows[:, 0], seconds, side='right')) - 1, 0), len(rows) - 1)
    row = rows[i]
    data.time = row[0]
    data.qpos[:] = row[1:1+model.nq]
    data.qvel[:] = row[1+model.nq:1+model.nq+model.nv]
    data.ctrl[:] = row[1+model.nq+model.nv:]
    mujoco.mj_forward(model, data)
    renderer = episode['renderer']
    renderer.update_scene(data, camera='torso_context')
    canvas = Image.new('RGB', (640, 640), '#101820')
    canvas.paste(Image.fromarray(renderer.render().copy()), (0, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=17)
    draw.text((14, 12), title, font=font, fill='white')
    result = episode['result']
    done = seconds >= rows[-1, 0]
    status = ('SUCCESS' if result['success'] else 'FAIL') if done else 'Executing recorded program'
    color = '#6ce5ac' if done and result['success'] else '#ffb478' if done else '#dae4eb'
    draw.text((14, 538), f't = {row[0]:.1f} s | {status}', font=font, fill=color)
    history = [h for h in result['history'] if h['started_s'] <= row[0] + 1e-6]
    if history:
        call = history[-1]
        code = f"{call['skill']}({call['index'] if call['index'] is not None else ''})"
        draw.text((14, 570), code, font=font, fill='white')
    if done:
        detail = (f"{result['fasteners']['intact']}/8 intact nails; hold {result['inverted_hold_s']:.1f}s"
                  if result['success'] else str(result['error'] or 'Strict acceptance failed'))
        draw.text((14, 604), detail[:72], font=small, fill=color)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--compare')
    parser.add_argument('--title', default='Recorded assembly episode')
    parser.add_argument('--compare-title', default='After RL')
    parser.add_argument('--out', required=True, help='Output .mp4 path; adjacent .png poster and .json metadata')
    parser.add_argument('--speed', type=float, default=15.)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error('speed must be positive')
    episodes = [load(args.episode)] + ([load(args.compare)] if args.compare else [])
    titles = [args.title, args.compare_title]
    duration = max(e['rows'][-1, 0] for e in episodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=10, codec='libx264', quality=8, macro_block_size=16) as writer:
        for seconds in list(np.arange(0, duration, args.speed / 10)) + [duration] * 20:
            frame = Image.new('RGB', (640*len(episodes), 672), '#101820')
            for i, episode in enumerate(episodes):
                frame.paste(panel(episode, seconds, titles[i]), (640*i, 0))
            ImageDraw.Draw(frame).text((14, 648), f'{args.speed:g}x simulation speed | recorded MuJoCo states | paused side remains at its last recorded state',
                                      font=ImageFont.load_default(size=15), fill='#9eafba')
            writer.append_data(np.array(frame))
        frame.save(out.with_suffix('.png'))
    for episode in episodes:
        episode['renderer'].close()
    out.with_suffix('.json').write_text(json.dumps(dict(episodes=[args.episode, args.compare],
        titles=titles[:len(episodes)], speed=args.speed, simulation_duration_s=float(duration),
        states_replayed=True, dynamics_rerun=False,
        display_headlight=dict(ambient=[.4]*3, diffuse=[.7]*3)), indent=2))
    print(out)


if __name__ == '__main__':
    main()
