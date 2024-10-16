from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

dataset = load_dataset('jeggers/gpqa_formatted', 'main', split='train')

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model_name = "FacebookAI/roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset['options'][0]))

def preprocess_function(examples):
    inputs = [f"{q} {opt}" for q, opt in zip(examples['Question'], examples['options'])]
    model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=512)
    model_inputs['labels'] = examples['answer']  # Set labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Recursive CoT
def recursive_cot(model, inputs, question_id, initial_steps=1, confidence_threshold=0.85, max_steps=5, output_dir='./reasoning_steps'):
    
    output_sequences = []
    current_step = initial_steps
    previous_reasoning = ""

    while current_step <= max_steps:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            confidence_scores = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)

        output_sequences.append((pred.cpu().numpy(), confidence_scores.cpu().numpy()))

        max_confidence = confidence_scores.max()

        # check confidence
        if max_confidence > confidence_threshold:
            break
        
        reasoning = f"Step {current_step}: I predict that option {pred.cpu().numpy()[0]} is the best choice with confidence {max_confidence:.2f}."
        previous_reasoning += reasoning + " "

        new_input_text = f"{inputs['input_ids']} Reasoning so far: {previous_reasoning.strip()}"
        inputs = tokenizer(new_input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        current_step += 1 

    return output_sequences

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

#training args
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

#trainer
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

#train
trainer.train()

#eval
results = trainer.evaluate()
print("Evaluation results:", results)
print("Accuracy:", results['eval_accuracy'])
