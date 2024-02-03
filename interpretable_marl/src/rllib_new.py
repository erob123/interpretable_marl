from typing import Union, List
import gym
import os
import tempfile
import random
import numpy as np

import logging
from datetime import datetime

from .actions import ActionInterpretable
from .agents_new import AgentPairInterpretable

from sentence_transformers import SentenceTransformer

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.tune.result import DEFAULT_RESULTS_DIR

from human_aware_rl.rllib.utils import (
    get_base_ae,
    iterable_equal,
    softmax,
)
from overcooked_ai_py.mdp.actions import Action
from human_aware_rl.rllib.rllib import (
    OvercookedMultiAgent,
    RlLibAgent,
    TrainingCallbacks
)

timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

class RlLibAgentInterpretable(RlLibAgent):
    """
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """

    def action(self, state, last_joint_action_descriptive):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """

        ## Modified
        # Preprocess the environment state
        obs = self.featurize(state, last_joint_action_descriptive)
        my_obs = obs[self.agent_index]

        # Use Rllib.Policy class to compute action argmax and action probabilities

        ## Modified from original since my_obs is a dict now, and we are using _disable_preprocessor_api
        _, rnn_state, info = self.policy.compute_actions(
            my_obs, self.rnn_state
        )
        ###############

        # Softmax in numpy to convert logits to normalized probabilities
        logits = info["action_dist_inputs"]
        action_probabilities = softmax(logits)

        # The original design is stochastic across different games,
        # Though if we are reloading from a checkpoint it would inherit the seed at that point, producing deterministic results
        [action_idx] = random.choices(
            list(range(Action.NUM_ACTIONS)), action_probabilities[0]
        )
        agent_action = Action.INDEX_TO_ACTION[action_idx]

        agent_action_info = {"action_probs": action_probabilities}
        self.rnn_state = rnn_state

        return agent_action, agent_action_info

class OvercookedMultiAgentInterpretable(OvercookedMultiAgent):
    """
    Class used to add in embeddings about the action chosen into the environment observation
    for interpretable MARL
    """

    def __init__(
        self,
        base_env,
        reward_shaping_factor=0.0,
        reward_shaping_horizon=0,
        bc_schedule=None,
        use_phi=True,
        ACTION_DESCRIPTION_EMBEDDING_NAME = "all-MiniLM-L6-v2"
    ):

        # Load model and tokenizer
        self.action_embedding_model = SentenceTransformer(ACTION_DESCRIPTION_EMBEDDING_NAME)

        super(OvercookedMultiAgentInterpretable, self).__init__(
            base_env, reward_shaping_factor, reward_shaping_horizon, bc_schedule, use_phi
        )

    def _embed_action_statements(self, action_statements: Union[str, List[str]]):
        return OvercookedMultiAgentInterpretable.embed_action_statements(
                action_statements=action_statements, 
                action_embedding_model=self.action_embedding_model
            )
    
    @classmethod
    def embed_action_statements(cls, action_statements: Union[str, List[str]], action_embedding_model: SentenceTransformer):

        single = False
        if type(action_statements) is str:
            single = True
            action_statements = [action_statements]

        embeddings = action_embedding_model.encode(action_statements)

        if single:
            return embeddings[0]

        return embeddings

    def _get_standard_start_action_embedding(self, num_agents):
        return tuple(self._embed_action_statements("No prior observations") for i in range(num_agents))

    def _setup_observation_space(self, agents):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        # ppo observation
        featurize_fn_ppo = (
            lambda state: self.base_env.lossless_state_encoding_mdp(state)
        )
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape

        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0

        ### MODIFIED FROM ORIGINAL OVERCOOKED AI #####
        primary_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )

        action_embedding_shape = self._get_standard_start_action_embedding(len(featurize_fn_ppo(dummy_state)))[0].shape
        high = np.ones(action_embedding_shape) * float("inf")
        low = np.ones(action_embedding_shape) * float("-inf")

        action_embedding_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )

        space_dict = {
            "primary": primary_observation_space,
            "action_embedding": action_embedding_observation_space
        }

        self.ppo_observation_space = gym.spaces.Dict(space_dict)
        ##############################################

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(
            state
        )
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )
        # hardcode mapping between action space and agent
        ob_space = {}
        for agent in agents:
            if agent.startswith("ppo"):
                ob_space[agent] = self.ppo_observation_space
            else:
                ob_space[agent] = self.bc_observation_space
        self.observation_space = gym.spaces.Dict(ob_space)

    def _get_featurize_fn(self, agent_id):
        return OvercookedMultiAgentInterpretable.get_featurize_fn(
                agent_id=agent_id, 
                env=self.base_env, 
                action_embedding_function=self._embed_action_statements
            )

    @classmethod
    def get_featurize_fn(cls, agent_id, env, action_embedding_function, batch=False):
        if agent_id.startswith("ppo"):
            ## Modified###
            def featurize(state, last_joint_action):

                # [0] is current agent, [1] is "other" agent for both
                cnn_state_encodings = env.lossless_state_encoding_mdp(state)
                action_embeddings = action_embedding_function(last_joint_action).astype(np.float32)

                # we want current agent state obs, or cnn_state_encodings[0], and other agent
                # action embedding, or action_embeddings[1]
                current_agent_state_dict = {
                    "primary": cnn_state_encodings[0].astype(np.float32),
                    "action_embedding": action_embeddings[1]
                }

                # now switch
                other_agent_state_dict = {
                    "primary": cnn_state_encodings[1].astype(np.float32),
                    "action_embedding": action_embeddings[0]
                }

                # add a batch dim if needed (for evaluation mostly)
                if batch:
                    for key in current_agent_state_dict.keys():
                        current_agent_state_dict[key] = np.expand_dims(current_agent_state_dict[key], axis=0)

                    for key in other_agent_state_dict.keys():
                        other_agent_state_dict[key] = np.expand_dims(other_agent_state_dict[key], axis=0)

                # return the list [curr, other]
                return [current_agent_state_dict, other_agent_state_dict]

            return featurize
            ###########
        if agent_id.startswith("bc"):
            def featurize(state, _):
                state_encodings = env.featurize_state_mdp(state)
                return state_encodings.astype(np.float32)
        raise ValueError("Unsupported agent type {0}".format(agent_id))

    def _get_obs(self, state, *, last_joint_action):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state, last_joint_action)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state, last_joint_action)[1]

        # modified to do the .astype(np.float32) type convertion within featurize
        # to account for new state dict setup
        return ob_p0, ob_p1

    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [
            action_dict[self.curr_agents[0]],
            action_dict[self.curr_agents[1]],
        ]

        assert all(
            self.action_space[agent].contains(action_dict[agent])
            for agent in action_dict
        ), "%r (%s) invalid" % (action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=True
            )
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=False
            )
            dense_reward = info["shaped_r_by_agent"]

        ob_p0, ob_p1 = self._get_obs(next_state, last_joint_action=ActionInterpretable.joint_action_to_char(joint_action))

        shaped_reward_p0 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[0]
        )
        shaped_reward_p1 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[1]
        )

        obs = {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
        rewards = {
            self.curr_agents[0]: shaped_reward_p0,
            self.curr_agents[1]: shaped_reward_p1,
        }
        dones = {
            self.curr_agents[0]: done,
            self.curr_agents[1]: done,
            "__all__": done,
        }
        infos = {self.curr_agents[0]: info, self.curr_agents[1]: info}
        return obs, rewards, dones, infos
    
    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state, last_joint_action=["No prior observations","No prior observations"])
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
    
    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgentInterpretable constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert (
            env_config
            and "env_params" in env_config
            and "multi_agent_params" in env_config
        )
        assert (
            "mdp_params" in env_config
            or "mdp_params_schedule_fn" in env_config
        ), "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(
            mdp_params, env_params, outer_shape, mdp_params_schedule_fn
        )
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)


# NO CHANGE FROM CORE OVERCOOKED, JUST DEFINING HERE SO THAT THE EVALUATE FUNCTION IS USED######
def get_rllib_eval_function(
    eval_params,
    eval_mdp_params,
    env_params,
    outer_shape,
    agent_0_policy_str="ppo",
    agent_1_policy_str="ppo",
    verbose=False,
):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    agent_1_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    Note: Agent policies are shuffled each time, so agent_0_policy_str and agent_1_policy_str are symmetric
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        if verbose:
            print("Computing rollout of current trained policy")

        # Randomize starting indices
        policies = [agent_0_policy_str, agent_1_policy_str]
        np.random.shuffle(policies)
        agent_0_policy, agent_1_policy = policies

        # Get the corresponding rllib policy objects for each policy string name
        agent_0_policy = trainer.get_policy(agent_0_policy)
        agent_1_policy = trainer.get_policy(agent_1_policy)

        agent_0_feat_fn = agent_1_feat_fn = None
        if "bc" in policies:
            base_ae = get_base_ae(eval_mdp_params, env_params)
            base_env = base_ae.env
            bc_featurize_fn = lambda state: base_env.featurize_state_mdp(state)
            if policies[0] == "bc":
                agent_0_feat_fn = bc_featurize_fn
            if policies[1] == "bc":
                agent_1_feat_fn = bc_featurize_fn

        # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this
        # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        results = evaluate(
            eval_params,
            eval_mdp_params,
            outer_shape,
            agent_0_policy,
            agent_1_policy,
            agent_0_feat_fn,
            agent_1_feat_fn,
            verbose=verbose,
        )

        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics["average_sparse_reward"] = np.mean(results["ep_returns"])
        return metrics

    return _evaluate
#######################################################################################


def evaluate(
    eval_params,
    mdp_params,
    outer_shape,
    agent_0_policy,
    agent_1_policy,
    agent_0_featurize_fn=None,
    agent_1_featurize_fn=None,
    verbose=False,
):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    if verbose:
        print("eval mdp params", mdp_params)
    evaluator = get_base_ae(
        mdp_params,
        {"horizon": eval_params["ep_length"], "num_mdp": 1},
        outer_shape,
    )

    ## Modified
    def action_embedding_function(action_statements):
        return OvercookedMultiAgentInterpretable.embed_action_statements(
                action_statements=action_statements, 
                action_embedding_model=SentenceTransformer(eval_params["ACTION_DESCRIPTION_EMBEDDING_NAME"])
            )


    ppo_default_featurize_fn = OvercookedMultiAgentInterpretable.get_featurize_fn(
            agent_id="ppo", 
            env=evaluator.env, 
            action_embedding_function=action_embedding_function,
            batch=True
        )

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = (
        agent_0_featurize_fn
        if agent_0_featurize_fn
        else ppo_default_featurize_fn
    )
    agent_1_featurize_fn = (
        agent_1_featurize_fn
        if agent_1_featurize_fn
        else ppo_default_featurize_fn
    )
    #####

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgentInterpretable(
        agent_0_policy, agent_index=0, featurize_fn=agent_0_featurize_fn
    )
    agent1 = RlLibAgentInterpretable(
        agent_1_policy, agent_index=1, featurize_fn=agent_1_featurize_fn
    )

    # Compute rollouts
    if "store_dir" not in eval_params:
        eval_params["store_dir"] = None
    if "display_phi" not in eval_params:
        eval_params["display_phi"] = False
    results = evaluator.evaluate_agent_pair(
        AgentPairInterpretable(agent0, agent1),
        num_games=eval_params["num_games"],
        display=eval_params["display"],
        dir=eval_params["store_dir"],
        display_phi=eval_params["display_phi"],
        info=verbose,
    )

    return results

###########################
# rllib.Trainer functions #
###########################


def gen_trainer_from_params(params):
    # All ray environment set-up
    if not ray.is_initialized():
        init_params = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_temp_dir": params["ray_params"]["temp_dir"],
            "log_to_driver": params["verbose"],
            "logging_level": logging.INFO
            if params["verbose"]
            else logging.CRITICAL
        }
        ray.init(**init_params)
    register_env("overcooked_multi_agent_interpretable", params["ray_params"]["env_creator"])
    ModelCatalog.register_custom_model(
        params["ray_params"]["custom_model_id"],
        params["ray_params"]["custom_model_cls"],
    )
    # Parse params
    model_params = params["model_params"]
    training_params = params["training_params"]
    environment_params = params["environment_params"]
    evaluation_params = params["evaluation_params"]
    bc_params = params["bc_params"]
    multi_agent_params = params["environment_params"]["multi_agent_params"]

    ## Modified to return the new environment
    env = OvercookedMultiAgentInterpretable.from_config(environment_params)
    ############################################

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"):
        # supported policy types thus far
        assert policy_type in ["ppo", "bc"]

        if policy_type == "ppo":
            config = {
                "model": {
                    "custom_model_config": model_params,
                    "custom_model": "MyPPOModel"
                },
                "_disable_preprocessor_api": True
            }
            return (
                None,
                env.ppo_observation_space,
                env.shared_action_space,
                config,
            )
        elif policy_type == "bc":
            bc_cls = bc_params["bc_policy_cls"]
            bc_config = bc_params["bc_config"]
            return (
                bc_cls,
                env.bc_observation_space,
                env.shared_action_space,
                bc_config,
            )

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = "{0}_{1}_{2}".format(
        params["experiment_name"], params["training_params"]["seed"], timestr
    )

    def custom_logger_creator(config):
        """Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>"""
        results_dir = params["results_dir"]
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                print(
                    "error creating custom logging dir. Falling back to default logdir {}".format(
                        DEFAULT_RESULTS_DIR
                    )
                )
                results_dir = DEFAULT_RESULTS_DIR
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=results_dir)
        logger = UnifiedLogger(config, logdir, loggers=None)
        return logger

    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {}
    all_policies = ["ppo"]

    # Whether both agents should be learned
    self_play = iterable_equal(
        multi_agent_params["bc_schedule"],
        OvercookedMultiAgent.self_play_bc_schedule,
    )
    if not self_play:
        all_policies.append("bc")

    multi_agent_config["policies"] = {
        policy: gen_policy(policy) for policy in all_policies
    }

    def select_policy(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("ppo"):
            return "ppo"
        if agent_id.startswith("bc"):
            return "bc"

    multi_agent_config["policy_mapping_fn"] = select_policy
    multi_agent_config["policies_to_train"] = {"ppo"}

    if "outer_shape" not in environment_params:
        environment_params["outer_shape"] = None

    if "mdp_params" in environment_params:
        environment_params["eval_mdp_params"] = environment_params[
            "mdp_params"
        ]
    
    # "framework": "tf2" and "eager_tracing": False if eager is desired
    trainer = PPOTrainer(
        env="overcooked_multi_agent_interpretable",
        config={
            "multiagent": multi_agent_config,
            "callbacks": TrainingCallbacks,
            "custom_eval_function": get_rllib_eval_function(
                evaluation_params,
                environment_params["eval_mdp_params"],
                environment_params["env_params"],
                environment_params["outer_shape"],
                "ppo",
                "ppo" if self_play else "bc",
                verbose=params["verbose"],
            ),
            "env_config": environment_params,
            "eager_tracing": False,
            "_disable_preprocessor_api": True,
            **training_params,
        },
        logger_creator=custom_logger_creator,
    )
    return trainer