def make_api_call(model, prompt):
    response = model.generate_response(prompt)
    return response