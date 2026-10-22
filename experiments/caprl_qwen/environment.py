"""Physics-verified code-as-policy: control a MuJoCo cube pusher."""
import ast
import numpy as np
import mujoco

XML = """
<mujoco>
<option timestep="0.005"/>
<worldbody>
<geom type="plane" size="1 1 0.1" friction="1 0.05 0.01"/>
<body name="cube" pos="0 0 0.025"><freejoint name="joint"/>
<geom type="box" size="0.025 0.025 0.025" mass="0.1" friction="1 0.05 0.01"/></body>
<body mocap="true" pos="-0.12 0 0.035"><geom type="sphere" size="0.027"/></body>
</worldbody></mujoco>
"""
DIR = {"east":(1,0),"west":(-1,0),"north":(0,1),"south":(0,-1)}
PROMPT = ('You write Python to control a tabletop robot. Available functions: '
          'push("east"), push("west"), push("north"), push("south"). '
          'Each pushes the cube approximately 10 cm. Output 1 or 2 '
          'push(...) calls only. Cube xy: ({:.2f}, {:.2f}). '
          'Target xy: ({:.2f}, {:.2f}).\nCode:\n')

def parse(code):
    tree=ast.parse(code.strip())
    if not 1 <= len(tree.body) <= 2: raise ValueError("call count")
    result=[]
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("invalid statement")
        c=node.value
        if not (isinstance(c.func, ast.Name) and c.func.id=="push" and
                len(c.args)==1 and not c.keywords and
                isinstance(c.args[0], ast.Constant) and
                isinstance(c.args[0].value,str) and c.args[0].value in DIR):
            raise ValueError("invalid call")
        result.append(c.args[0].value)
    return result

class Push:
    def __init__(self):
        self.m=mujoco.MjModel.from_xml_string(XML)
        self.adr=self.m.joint("joint").qposadr[0]
        self.bid=self.m.body("cube").id
    def reset(self,seed):
        self.d=mujoco.MjData(self.m)
        rng=np.random.default_rng(seed)
        self.start=rng.uniform(-.04,.04,2)
        axis=int(rng.integers(0,2))
        delta=np.zeros(2)
        delta[axis]=int(rng.choice([-1,1]))*.16
        self.goal=self.start+delta
        self.d.qpos[self.adr:self.adr+2]=self.start
        mujoco.mj_forward(self.m,self.d)
        return PROMPT.format(*self.start,*self.goal)
    def xy(self):
        return self.d.xpos[self.bid,:2].copy()
    def move(self,point,steps):
        begin=self.d.mocap_pos[0].copy()
        end=np.array([*point,.035])
        for k in range(steps):
            self.d.mocap_pos[0]=begin+(end-begin)*(k+1)/steps
            mujoco.mj_step(self.m,self.d)
    def push(self,name):
        v=np.asarray(DIR[name],float)
        center=self.xy()
        self.d.mocap_pos[0,2]=.3
        self.move(center-.1*v,80)
        self.d.mocap_pos[0,2]=.035
        self.move(center+.12*v,150)
    def execute(self,code):
        try:
            calls=parse(code)
        except (SyntaxError,ValueError,TypeError):
            return 0.,False,False
        start=float(np.linalg.norm(self.xy()-self.goal))
        for name in calls: self.push(name)
        end=float(np.linalg.norm(self.xy()-self.goal))
        success=end<.065
        return float(np.clip((start-end)/.16,-1,1)+success),True,bool(success)

def expert(seed):
    env=Push()
    prompt=env.reset(seed)
    delta=env.goal-env.start
    axis=int(np.argmax(np.abs(delta)))
    direction=("east" if delta[0]>0 else "west") if axis==0 else ("north" if delta[1]>0 else "south")
    return prompt,'push("'+direction+'")'

if __name__=="__main__":
    env=Push()
    for seed in range(8):
        prompt, code=expert(seed)
        env.reset(seed)
        print(seed,code,env.execute(code))
