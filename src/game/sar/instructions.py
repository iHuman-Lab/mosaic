from minigrid.envs.babyai.core.verifier import Instr

from .objects import REAL_VICTIMS


def calculate_max_steps(
    room_size: int,
    num_rows: int,
    num_cols: int,
    num_doors: int,
    victims_per_room: int,
    human_exploration_factor: float = 0.5,
    steps_per_victim: int = 5,
    safety_buffer: float = 1.0,
):
    """
    Calculate a human-friendly max step limit for a MiniGrid-style environment.

    Parameters
    ----------
    room_size : int
        Size of one room (room_size x room_size).
    grid_rows : int
        Number of room rows.
    grid_cols : int
        Number of room columns.
    num_doors : int
        Total number of doors in the maze.
    victims_per_room : int
        Number of real victims per room.
    human_exploration_factor : float
        Multiplier for human inefficiency (default = 2.0).
    steps_per_victim : int
        Extra steps to approach + rescue one victim.
    safety_buffer : float
        Final multiplicative slack for mistakes and hesitation.

    Returns
    -------
    int
        Recommended max_steps.
    """

    num_rooms = num_rows * num_cols

    # 1. Human exploration cost per room
    nav_time_room = human_exploration_factor * (room_size**2)

    # 2. Base exploration of all rooms
    exploration_cost = num_rooms * nav_time_room

    # 3. Door & key backtracking cost
    # Each door ≈ one extra room traversal
    door_cost = num_doors * nav_time_room

    # 4. Victim rescue cost
    total_victims = num_rooms * victims_per_room
    victim_cost = total_victims * steps_per_victim

    # 5. Total before buffer
    raw_steps = exploration_cost + door_cost + victim_cost

    # 6. Safety buffer for human mistakes
    max_steps = int(raw_steps * safety_buffer)

    return max_steps


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
        self.victims = victims
        self.victim_types = [type(v) for v in victims]
        self.num_victims = len(victims)
        self.env = None
        self.start_room = None

    def reset_verifier(self, env):
        """Initialize verifier with environment context.

        We record the starting room so that completion requires the agent
        to leave that room (e.g., open the door and enter the next room)
        after rescuing victims.
        """
        self.env = env
        try:
            self.start_room = env.room_from_pos(*env.agent_pos)
        except Exception:
            self.start_room = None

    def verify(self, *args, **kwargs):
        """Verify that all victims are picked and agent has left start room.

        Returns 'success' only when there are no remaining real victims and
        the agent is located in a different room than where it started.
        Otherwise returns 'continue'.
        """
        if self.env is None:
            return "continue"

        # Use utility method to count remaining victims
        remaining_victims = self.env._count_objects_by_type(REAL_VICTIMS)

        if remaining_victims > 0:
            return "continue"

        # If we have a recorded start room, require agent to have left it
        if self.start_room is not None:
            try:
                current_room = self.env.room_from_pos(*self.env.agent_pos)
                if current_room is self.start_room:
                    return "continue"
            except Exception:
                # If room computation fails, fall back to simple success
                pass

        return "success"

    def surface(self, env):
        """
        Return a natural language description of the instruction.

        Args:
            env: The environment instance

        Returns:
            str: Description of the instruction
        """
        return f"pick up all {self.num_victims} victims"
