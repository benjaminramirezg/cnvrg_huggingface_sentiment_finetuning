from transformers import BertTokenizerFast
from datasets import load_dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

#########
# Model #
#########

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)
for param in model.base_model.parameters():
  param.requires_grad = False

###########
# Dataset #
###########

def encode(examples):
  return tokenizer(examples['review_body'], truncation=True, padding='max_length')

def encode_labels(example):
  example['labels'] = example['stars'] - 1
  return example

train_dataset = load_dataset("amazon_reviews_multi", "es", split="train")
test_dataset = load_dataset("amazon_reviews_multi", "es", split="test")

train_dataset = train_dataset.map(encode, batched=True)
test_dataset = test_dataset.map(encode, batched=True)

train_dataset = train_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

############
# Training #
############

training_args = TrainingArguments(
    output_dir='./output/',              # output directory
    num_train_epochs=1,                  # total # of training epochs
    per_device_train_batch_size=16,      # batch size per device during training
    per_device_eval_batch_size=64,       # batch size for evaluation
    warmup_steps=500,                    # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # strength of weight decay
    logging_dir='./logs/',               # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

########
# Main #
########

if __name__ == "__main__":
    trainer.train()
    trainer.save_model()
    trainer.evaluate()
