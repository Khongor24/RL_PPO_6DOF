import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("my_robot.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)

        for _ in range(10):
            mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.02)