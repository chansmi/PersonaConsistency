import json
from src.utils.data_utils import load_json
from src.utils.api_utils import make_api_call

class BaseExperiment:
    def __init__(self, model):
        self.model = model

class MachIVExperiment(BaseExperiment):
    def __init__(self, model, persona_prompt=None):
        super().__init__(model)
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
        output_file = "results/vanilla_model/mach_iv_scores.json"
        if self.persona_prompt:
            output_file = "results/persona_model/mach_iv_scores.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f)
