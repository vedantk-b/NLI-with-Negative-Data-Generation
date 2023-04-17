from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

max_length = 256
hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

from data_prep import get_train_eval_data

train_data, eval_data = get_train_eval_data()

tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name).to(device)

def preprocess_training_examples(examples):
    inputs = tokenizer(examples["text"], examples["reason"], max_length = 256, truncation=True, padding = "max_length", return_token_type_ids=True)
    inputs["labels"] = examples["label"]
    return inputs

hg_validation_dataset = eval_data.map(preprocess_training_examples, batched = True, remove_columns = eval_data.column_names)
hg_validation_dataset

from torch.utils.data import DataLoader
from transformers import default_data_collator

eval_dataloader = DataLoader(
    hg_validation_dataset, collate_fn=default_data_collator, batch_size=8
)

outlis = np.array([])
problis = np.array([])
# model = model.to(device)
for batch in eval_dataloader:
  input_ids = batch["input_ids"].to(device)
  attention_mask = batch["attention_mask"].to(device)
  token_type_ids = batch["token_type_ids"].to(device)
  outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
  
  predicted_probability = torch.softmax(outputs.logits, dim=1).to("cpu").detach()  # batch_size only one
  # print(predicted_probability)
  # print(outputs)

  problis = (np.append(problis,np.array(predicted_probability[:, 0]), axis = 0))
  outlis = np.append(outlis, batch["labels"].detach().to("cpu"))
  # break
  # preds = out.logits.argmax(dim = 1)
  # loss = out.loss
  # pos_probs = softmax(out.logits.to("cpu").detach(), axis=1)
  # problis = (np.append(problis,np.array(pos_probs[:, 1]), axis = 0))
  # outlis = np.append(outlis, batch["labels"].detach().to("cpu"))
  

from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_predictions(outlis, problis, name="roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
_ = display.ax_.set_title("2-class Precision-Recall curve")

predictions = [int(i > 0.5) for i in problis]
labels = outlis

tp, fp, tn, fn = 0, 0, 0, 0
for i in range(len(predictions)):
  if(labels[i] == 1):
    if(predictions[i] == labels[i]):
      tp += 1
    else:
      fp += 1
  if(labels[i] == 0):
    if(predictions[i] == labels[i]):
      tn += 1
    else:
      fn += 1
confusion_matrix = [[tp, fn], [fp, tn]]
p = tp/(tp + fp)
r = tp/(tp + fn)
f1 = 2*p*r/(p + r)

print(f"Confusion Matrix: \n {confusion_matrix[0]}\n    {confusion_matrix[1]} \n\n Precision: {p} \n Recall: {r} \n f1-Score: {f1}")