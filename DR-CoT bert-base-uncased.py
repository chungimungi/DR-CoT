from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback
import torch
from sklearn.metrics import accuracy_score
import os

dataset = load_dataset('jeggers/gpqa_formatted', 'main', split='train')

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset['options'][0]))

def preprocess_function(examples):
    inputs = [f"{q} {opt}" for q, opt in zip(examples['Question'], examples['options'])]
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=512)
    model_inputs['labels'] = examples['answer'] 
    return model_inputs
    
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# DR-CoT: Dynamic Recursive CoT with Meta-Reasoning
def dynamic_recursive_cot(model, tokenizer, inputs, question_id, current_step=1, confidence_threshold=0.85, 
                          max_steps=10, previous_reasoning="", output_sequences=None, confidence_scores=None, 
                          branching_factor=2):
    
    if output_sequences is None:
        output_sequences = []
    if confidence_scores is None:
        confidence_scores = []

    # Cache key (not used but prepared for future reasoning caching)
    cache_key = (tuple(inputs['input_ids'].cpu().numpy()), previous_reasoning)

    # Terminate if the step limit is exceeded
    if current_step > max_steps:
        return output_sequences

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        new_confidence_scores = torch.softmax(logits, dim=-1)
        top_preds = logits.argsort(dim=-1, descending=True)[0, :branching_factor]

    # Append current predictions and confidence scores to output
    output_sequences.append((top_preds.cpu().numpy(), new_confidence_scores.cpu().numpy()))
    max_confidence = new_confidence_scores.max().item()
    confidence_scores.append(max_confidence)

    # Meta-reasoning: Stop if confidence is above the threshold
    if max_confidence > confidence_threshold:
        return output_sequences

    # Loop through multiple reasoning paths (branching factor)
    for pred in top_preds:
        reasoning_path = f"Step {current_step}: I predict option {pred.item()} with confidence {new_confidence_scores[0, pred].item():.2f}."
        new_reasoning = previous_reasoning + reasoning_path + " "
        
        # Prepare new inputs with the updated reasoning context
        new_input_text = f"{inputs['input_ids']} Reasoning so far: {new_reasoning.strip()}"
        new_inputs = tokenizer(new_input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        # Recursive call for each reasoning path
        new_output_sequences = dynamic_recursive_cot(
            model, tokenizer, new_inputs, question_id, current_step + 1, confidence_threshold,
            max_steps, new_reasoning, output_sequences, confidence_scores, branching_factor
        )
        
        output_sequences.append(new_output_sequences)

    return output_sequences

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

#train args
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",  
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-6,  
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01, 
    load_best_model_at_end=True,
    greater_is_better=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",
)

#lrs
num_training_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
warmup_steps = num_training_steps // 10
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=num_training_steps
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  
)

trainer.train()

#eval
results = trainer.evaluate()
print("Evaluation results:", results)
print("Accuracy:", results['eval_accuracy'])
