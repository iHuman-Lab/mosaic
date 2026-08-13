from minigrid.envs.babyai.core.verifier import Instr

from .objects import REAL_VICTIMS


class PickupAllVictimsInstr(Instr):
    """
    Instruction to pick up all victims in the environment.
    This instruction verifies that all victims have been picked up (removed from grid).
    """

    def __init__(self, victims):
        """
        Initialize instruction with list of victim objects to pick up.

        Args:
            victims: List of victim objects that need to be picked up
        """
        self.num_victims = len(victims)

    def reset_verifier(self, env):
        """Called on env reset; sets self.env so verify() can access the grid."""
        self.env = env

    def verify(self, action):
        """
        Verify if all victims have been picked up.

        Returns:
            str: 'success' if all victims picked up, 'continue' otherwise
        """
        remaining_victims = self.env.count_objects_by_type(REAL_VICTIMS)
        if remaining_victims == 0:
            return "success"
        return "continue"

    def surface(self, env):
        """
        Return a natural language description of the instruction.

        Args:
            env: The environment instance

        Returns:
            str: Description of the instruction
        """
        return f"pick up all {self.num_victims} victims"
