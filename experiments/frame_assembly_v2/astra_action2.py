import sys,json,subprocess
poses=json.load(open('outputs/frame_assembly/astra/grasp_targets.json'))
pose,x,y,z,seconds=sys.argv[1:6]
a={'seconds':float(seconds),'wrist_position':list(map(float,[x,y,z])),'wrist_quaternion':[.5,-.5,.5,-.5],'finger_angles':poses[pose],'reason':' '.join(sys.argv[6:])}
p=subprocess.run([sys.executable,'-m','experiments.frame_assembly_v2.agent_control','act','--trial','trial02','--json',json.dumps(a)],capture_output=True,text=True)
if p.returncode:print(p.stderr);sys.exit(p.returncode)
o=json.loads(p.stdout)
print(json.dumps({k:o[k] for k in ['time','wrist_position','fingertip_positions','board_poses','contact_normal_forces_N','hand_board_contacts','overview_image','hand_image']}))
