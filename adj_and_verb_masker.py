from flair.data import Sentence
from flair.models import SequenceTagger


# Create a function to replace adjectives and verbs
def replace_adj_and_verb(sentence, replacement):
    # Load the Part-of-Speech (POS) tagging pipeline
    pos_tagger = SequenceTagger.load("flair/pos-english")

    # Tag the parts of speech in the input sentence
    sent = Sentence(sentence)
    pos_tagger.predict(sent)

    # Create a list to store the modified tokens
    modified_tokens = []

    # Loop through each token and its corresponding POS tag
    i = 1
    while (sent.get_token(i)):
        chk = 0
        for pos_tag in ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            if pos_tag in str(sent.get_token(i)):
                chk = 1
        if(chk):
            modified_tokens.append(replacement)
        else:
            modified_tokens.append(list(str(sent.get_token(i)).split('"'))[1])
        i += 1

    # Join the modified tokens back into a sentence
    modified_sentence = " ".join(modified_tokens)

    return modified_sentence