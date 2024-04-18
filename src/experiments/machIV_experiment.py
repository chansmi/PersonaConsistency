import json
import os
from src.utils.data_utils import load_json
from src.utils.api_utils import make_api_call

class MachIVExperiment:
    def __init__(self, model, persona_prompt=None):
        self.model = model
        self.persona_prompt = persona_prompt
        self.questions = load_json("src/data/mach_iv_questions.json")

    def run(self):
        scores = {}
        for question_id, question_text in self.questions.items():
            prompt = self._build_prompt(question_text)
            response = make_api_call(self.model, prompt)
            log_probs = self._extract_log_probs(response)
            scores[question_id] = log_probs
        self.results = scores
        self.save_results()

    def _build_prompt(self, question_text):
        if self.persona_prompt:
            prompt = f"{self.persona_prompt}\n{question_text}"
        else:
            prompt = question_text
        return prompt

    def _extract_log_probs(self, response):
        log_probs = {}
        for score in range(1, 6):
            log_probs[score] = response.get_log_prob(str(score))
        return log_probs

    def save_results(self):
        persona_dir = self.persona_prompt.replace(" ", "_").lower() if self.persona_prompt else "vanilla"
        model_dir = self.model.model_name.replace(" ", "_").lower()
        
        output_dir = f"results/{model_dir}/{persona_dir}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/mach_iv_scores.json"
        version = 1
        # Check if the file exists and iterate version numbers if it does
        while os.path.exists(f"{output_file[:-5]}_v{version}.json"):
            version += 1
        output_file = f"{output_file[:-5]}_v{version}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=4)
