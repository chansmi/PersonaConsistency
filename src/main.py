import argparse
from backend import model_backend_factory
from experiments.mach_iv_experiment import MachIVExperiment

def main(model_name, persona_prompt=None):
    # Initialize the appropriate backend for the given model
    model_backend = model_backend_factory(model_name)

    # Initialize and run the MachIVExperiment
    experiment = MachIVExperiment(model_backend, persona_prompt)
    experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mach IV Experiment with specified model and optional persona.")
    parser.add_argument("model_name", type=str, help="The name of the model to use for the experiment.")
    parser.add_argument("--persona_prompt", type=str, default=None, help="Optional persona prompt to be used in the experiment.")
    
    args = parser.parse_args()
    
    main(args.model_name, args.persona_prompt)
