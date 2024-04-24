import json
import os
from utils.data_utils import load_json
from utils.api_utils import make_api_call

class MachIVExperiment:
    def __init__(self, model, persona_prompt=None):
        self.model = model
        self.persona_prompt = persona_prompt
        self.questions = load_json("data/mach_iv_questions.json")

    def run(self):
        scores = {}
        for question_id, question_text in self.questions.items():
            prompt = self._build_prompt(question_text)
            response = make_api_call(self.model, prompt)
            selected_score = self._extract_selected_score(response)
            scores[question_id] = {"selected_score": selected_score}
        self.results = scores
        self.save_results()

    def _build_prompt(self, question_text):
        scale = "0 = Disagree, 1 = Slightly Disagree, 2 = Neutral, 3 = Slightly Agree, 4 = Agree"
        if self.persona_prompt:
            prompt = f"{self.persona_prompt}\n{question_text}\nPlease select your response from the following scale:\n{scale}"
        else:
            prompt = f"{question_text}\nPlease select your response from the following scale:\n{scale}"
        return prompt

    def _extract_selected_score(self, response):
        try:
            parts = [part for part in response.replace('=', ' ').split() if part]
            for part in parts:
                if part.isdigit():  # Check if the part is a number
                    selected_score = int(part)
                    if 0 <= selected_score <= 4:
                        return selected_score
                    else:
                        break 
        except ValueError:
            pass  
        print(f"Invalid score: {response}. Defaulting to 2.")
        return 2


def save_results(self):
    model_dir = self.model.model_name.replace(" ", "_").lower()
    output_dir = f"../results/{model_dir}" 
    os.makedirs(output_dir, exist_ok=True)  
    output_file = f"{output_dir}/mach_iv_scores.json"
    version = 1
    while os.path.exists(f"{output_file[:-5]}_v{version}.json"):
        version += 1
    output_file = f"{output_file[:-5]}_v{version}.json"  
    with open(output_file, "w") as f:
        json.dump(self.results, f, indent=4)
