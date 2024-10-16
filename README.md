# RecursiveCoT

The `recursive_cot` function generates reasoning iteratively, asking the model to predict an answer and assess its confidence at each step. If the model's confidence exceeds a set threshold, the process stops; otherwise, the reasoning is updated and fed back into the model as input for the next step. This continues until the model is confident enough or the maximum allowed steps are reached. Each step's prediction and reasoning are saved.
