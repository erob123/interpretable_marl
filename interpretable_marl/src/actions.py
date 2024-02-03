from overcooked_ai_py.mdp.actions import Action, Direction
import itertools

class ActionInterpretable(Action):
    """
    The six actions available in the OvercookedGridworld.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """

    STAY = (0, 0)
    INTERACT = "interact"
    ALL_ACTIONS = INDEX_TO_ACTION = Direction.INDEX_TO_DIRECTION + [
        STAY,
        INTERACT,
    ]
    INDEX_TO_ACTION_INDEX_PAIRS = [
        v for v in itertools.product(range(len(INDEX_TO_ACTION)), repeat=2)
    ]
    ACTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_ACTION)}
    MOTION_ACTIONS = Direction.ALL_DIRECTIONS + [STAY]

    ## modified for embedding
    ACTION_TO_CHAR = {
        Direction.NORTH: "I am moving North",
        Direction.SOUTH: "I am moving South",
        Direction.EAST: "I am moving East",
        Direction.WEST: "I am moving West",
        STAY: "I am staying still",
        INTERACT: "I am interacting with the object in front of me",
    }
    ######

    NUM_ACTIONS = len(ALL_ACTIONS)

    @staticmethod
    def to_char(action):
        assert action in ActionInterpretable.ALL_ACTIONS
        return ActionInterpretable.ACTION_TO_CHAR[action]

    # modified to return list instead of tuple
    @staticmethod
    def joint_action_to_char(joint_action):
        assert all([a in ActionInterpretable.ALL_ACTIONS for a in joint_action])
        return [ActionInterpretable.to_char(a) for a in joint_action]