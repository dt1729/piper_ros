"""
Adapter layer providing a mujoco_py-compatible API over modern mujoco.
"""
import mujoco
import mujoco.viewer as _viewer


load_model_from_path = mujoco.MjModel.from_xml_path


class MjModel:
    """Adapter: wraps modern MjModel to add mujoco_py-style joint_name2id."""

    def __init__(self, raw_model):
        self._raw = raw_model

    def joint_name2id(self, name):
        return mujoco.mj_name2id(self._raw, mujoco.mjtObj.mjOBJ_JOINT, name)

    def __getattr__(self, name):
        return getattr(self._raw, name)


class MjSim:
    """Adapter: wraps modern MjModel + MjData to behave like mujoco_py MjSim."""

    def __init__(self, raw_model):
        self._raw_model = raw_model
        self.model = MjModel(raw_model)
        self.data = mujoco.MjData(raw_model)

    def step(self):
        mujoco.mj_step(self._raw_model, self.data)


class MjViewer:
    """Adapter: wraps modern passive viewer to behave like mujoco_py MjViewer."""

    def __init__(self, sim):
        self._handle = _viewer.launch_passive(sim._raw_model, sim.data)

    def render(self):
        self._handle.sync()

    def is_running(self):
        return self._handle.is_running()
