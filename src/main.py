from src.models.gpt3 import GPT3Model
from src.experiments.mach_iv_experiment import MachIVExperiment

def main():
    # Run MACH-IV experiment with vanilla model
    vanilla_model = GPT3Model()
    vanilla_experiment = MachIVExperiment(vanilla_model)
    vanilla_experiment.run()

    # Run MACH-IV experiment with persona development
    persona_prompt = load_json("src/data/persona_prompts.json")["machiavellian"]
    persona_model = GPT3Model()
    persona_experiment = MachIVExperiment(persona_model, persona_prompt)
    persona_experiment.run()

if __name__ == "__main__":
    main()