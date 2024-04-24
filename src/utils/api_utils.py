def make_api_call(model, prompt):
    response = model.generate(prompt)
    return response