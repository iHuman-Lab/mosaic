from minigrid.envs.babyai.core.verifier import Instr

from .objects import REAL_VICTIMS


class PickupAllVictimsInstr(Instr):
    """
    Instruction to pick up all victims in the environment.
    This instruction verifies that all victims have been picked up (removed from grid).
    """

    def reset_verifier(self, env):
        """Called on env reset, once this episode's victims are placed; sets
        self.env so verify() can access the grid, and reads this episode's
        victim count off env.total_victims for num_navs_needed(). Note:
        surface() can't rely on this — RoomGridLevel._gen_grid() calls
        surface() before RoomGridLevel.reset() calls this — so it reads
        env.total_victims directly instead.
        """
        self.env = env
        self.num_victims = env.total_victims

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
        return f"pick up all {env.total_victims} victims"
