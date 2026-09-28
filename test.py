import time
import numpy as np
import mujoco
import mujoco.viewer

scene_path = "Simulation/SO101/scene.xml"
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

n_actuators = model.nu
low = model.actuator_ctrlrange[:, 0]
high = model.actuator_ctrlrange[:, 1]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        action = np.random.uniform(low, high, size=n_actuators)
        data.ctrl[:] = action

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))