
## Getting Started
1. Create conda environment: `conda create -n interpretable_marl python=3.7`
1. Activate conda environment: `conda activate interpretable_marl`
1. Install GPU deps: `conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0` (https://gretel.ai/blog/install-tensorflow-with-cuda-cdnn-and-gpu-support-in-4-easy-steps)
1. Save library path for gpu deps:

    ```
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```
1. Close and reopen terminal or SSH and reactivate conda environment if needed
1. Move into the root project directory: `cd interpretable_marl` (not `interpretable_marl/interpretable_marl`)
1. Clone overcooked_ai: `git clone https://github.com/HumanCompatibleAI/overcooked_ai.git` (not using git submodules since only one repo,
but if we add more, we should use submodules)
1. Install overcooked : `pip install -e overcooked_ai/`
1. Install poetry: `pip install poetry`
1. Install the root project with poetry: `poetry install`
1. Open & run the notebook tutorial in overcooked to verify the upstream overcooked setup works

## Running interpretable MARL
1. Login to wandb: `wandb login`
1. Run `python interpretable_marl/run_single.py` to verify the interpretable_marl setup works
1. Edit or run experiments as desired

TODO: Add instructions for editing model, augmenting state in both training and evaluation, running experiments,
changing action verbiage, debugging, changing config, etc.
