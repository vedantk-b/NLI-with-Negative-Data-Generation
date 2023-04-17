import torch
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

from data_prep import get_train_eval_data

train_data, eval_data = get_train_eval_data()

def preprocess_training_examples(examples):
    inputs = tokenizer(examples["text"], examples["reason"], max_length = 256, truncation=True, padding = "max_length")
    inputs["labels"] = examples["label"]
    return inputs

train_dataset = train_data.map(preprocess_training_examples, batched = True, remove_columns = train_data.column_names)
validation_dataset = eval_data.map(preprocess_training_examples, batched = True, remove_columns = eval_data.column_names)

from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataset.set_format("torch")
validation_dataset.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle = True,
    collate_fn = default_data_collator, batch_size = 16
)
eval_dataloader = DataLoader(
    validation_dataset, collate_fn=default_data_collator, batch_size=16
)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr = 2e-5)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = "cuda"
model = model.to(device)

from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


    torch.save(model.state_dict(), '/content/drive/MyDrive/Enterpret/wts.pt')
