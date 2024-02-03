from ppo_rllib_client_interpretable_marl import ex

def main():
    config_updates = {
        "seeds": [0],
        "layout_name": "coordination_ring",
        "clip_param": 0.069,
        "gamma": 0.975,
        "grad_clip": 0.359,
        "kl_coeff": 0.156,
        "lmbda": 0.5,
        "lr": 1.6e-4,
        "num_training_iters": 650,
        "old_dynamics": True,
        "reward_shaping_horizon": 5000000,
        "use_phi": False,
        "vf_loss_coeff": 9.33e-3,
        "results_dir": "reproduced_results/ppo_sp_coordination_ring"
    }

    run = ex.run(config_updates=config_updates, options={"--loglevel": "ERROR"})

if __name__ == "__main__":
    main()
