from ray.rllib.utils.annotations import override
from interpretable_marl.src.actions import ActionInterpretable
from overcooked_ai.src.overcooked_ai_py.agents.agent import AgentGroup

class AgentGroupInterpretable(AgentGroup):
    """
    AgentGroup is a group of N agents used to sample
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        super(AgentGroupInterpretable, self).__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)

        # defined as last actions taken by each agent indexed by agent_index
        self.last_joint_action = ["No prior observations","No prior observations"]

        # avoid circular imports
        from src.rllib_new import RlLibAgentInterpretable
        self.interpretable_cls = RlLibAgentInterpretable

    # modified to take in last_joint_action
    @override(AgentGroup)
    def joint_action(self, state):
        
        actions_and_probs_n = tuple(
            a.action(state, self.last_joint_action) 
            if type(a) is self.interpretable_cls 
            else a.action(state) 
            for a in self.agents
        )

        a_t, _ = zip(*actions_and_probs_n)
        self.last_joint_action = ActionInterpretable.joint_action_to_char(a_t)
        return actions_and_probs_n

    @override(AgentGroup)
    def reset(self):
        self.last_joint_action = ["No prior observations","No prior observations"]
        super(AgentGroupInterpretable, self).reset()

### This is copy/paste from original but just changing inheritance structure (AgentGroupInterpretable)
class AgentPairInterpretable(AgentGroupInterpretable):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        super().__init__(
            *agents, allow_duplicate_agents=allow_duplicate_agents
        )
        assert self.n == 2
        self.a0, self.a1 = self.agents

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play,
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_and_infos_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_and_infos_1 = self.a1.action(state)
            joint_action_and_infos = (action_and_infos_0, action_and_infos_1)
            return joint_action_and_infos
        else:
            return super().joint_action(state)