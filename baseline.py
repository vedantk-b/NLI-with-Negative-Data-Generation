from sentence_transformers import SentenceTransformer, util
from data_prep import get_data
import torch

train_df, eval_df = get_data()

sentences = ["I'm happy", "The app is crashing"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

util.pytorch_cos_sim(embedding_1, embedding_2)
## tensor([[0.6003]])

tx_embs = model.encode(eval_df["text"], convert_to_tensor = True).to("cpu")
rs_embs = model.encode(eval_df["reason"], convert_to_tensor = True).to("cpu")

out = util.pytorch_cos_sim(tx_embs, rs_embs)
pred = []
log_probs = []
for i, j in zip(range(len(eval_df)), range(len(eval_df))):
    pred.append(1 if out[i][j] > 0.5 else 0)
    log_probs.append(out[i][j])

labels = eval_df['label']

tp, fp, tn, fn = 0, 0, 0, 0
for i in range(len(pred)):
  if(labels[i] == 1):
    if(pred[i] == labels[i]):
      tp += 1
    else:
      fp += 1
  if(labels[i] == 0):
    if(pred[i] == labels[i]):
      tn += 1
    else:
      fn += 1
confusion_matrix = [[tp, fn], [fp, tn]]
p = tp/(tp + fp)
r = tp/(tp + fn)
f1 = 2*p*r/(p + r)

print(f"Confusion Matrix: \n {confusion_matrix[0]}\n    {confusion_matrix[1]} \n\n Precision: {p} \n Recall: {r} \n f1-Score: {f1}")

from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_predictions(labels, log_probs, name="all-mpnet-base-v2")
_ = display.ax_.set_title("2-class Precision-Recall curve")

    