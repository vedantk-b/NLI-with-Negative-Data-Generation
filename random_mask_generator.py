from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import AlbertTokenizer, AlbertModel


# Load the pre-trained DistilBERT tokenizer and model
rdntokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

from transformers import pipeline
import random

# Define the input sentence
input_sentence = "The quick brown fox jumps over the lazy dog."

# Create a function to randomly mask words
def mask_random_words(sentence, mask_token="[MASK]", mask_prob=0.2):
    # Tokenize the input sentence
    tokens = rdntokenizer(sentence, return_tensors="pt")

    # Get the token IDs and attention mask
    token_ids = tokens["input_ids"].squeeze().tolist()
    attention_mask = tokens["attention_mask"].squeeze().tolist()

    # Loop through each token and its corresponding attention mask
    for i in range(len(token_ids)):
        if random.random() < mask_prob and attention_mask[i] == 1:
            # If the random probability is less than mask_prob and the token is not a special token,
            # replace it with the mask_token
            token_ids[i] = rdntokenizer.convert_tokens_to_ids(mask_token)

    # Convert the masked token IDs back to tokens
    masked_tokens = rdntokenizer.convert_ids_to_tokens(token_ids)

    # Join the masked tokens back into a sentence
    masked_sentence = " ".join(masked_tokens)

    return masked_sentence

if __name__ == "__main__":
    # Call the function with the input sentence
    masked_sentence = mask_random_words(input_sentence)

    # Print the masked sentence
    print("Input Sentence: ", input_sentence)
    print("Masked Sentence: ", masked_sentence)