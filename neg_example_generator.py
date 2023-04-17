import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import AlbertTokenizer, AlbertModel


def neg_example_generator(input_sentence, tokenizer, model):
# Tokenize the input sentence
    tokens = tokenizer.encode(input_sentence, return_tensors="pt")

    # Find the positions of the masked tokens
    masked_tokens = tokens.clone()

    # Generate predictions for the masked token
    with torch.no_grad():
        outputs = model(masked_tokens)
        predictions = outputs.pooler_output.argmax(dim=-1)

    # Convert the predicted token ID back to a token
    predicted_token = tokenizer.convert_ids_to_tokens(predictions.squeeze().tolist())

    # Print the input sentence with the filled masks
    return (" ".join(predicted_token).replace(".", "")) 


tokenizer = AlbertTokenizer.from_pretrained(' ')
model = AlbertModel.from_pretrained("albert-base-v2")

from data_prep import get_data
from adj_and_verb_masker import replace_adj_and_verb
from random_mask_generator import mask_random_words

train_df, eval_df = get_data()

neglis = []
for text in train_df["text"]:
    text = text + "."
    masked_text = replace_adj_and_verb(text, '[MASK]')
    if( '[MASK]' not in masked_text):
        masked_text = mask_random_words(text, mask_token="[MASK]", mask_prob=0.2)
    output = neg_example_generator(masked_text)

negres = []
for text in train_df["reason"]:
    text = text + "."
    masked_text = replace_adj_and_verb(text, '[MASK]')
    if( '[MASK]' not in masked_text):
        masked_text = mask_random_words(text, mask_token="[MASK]", mask_prob=0.2)
    neg_text = neg_example_generator(masked_text)
    negres.append(neg_text)

file = open('neglis.txt','w')
for neg in neglis:
	file.write(neg+",\n")
file.close()