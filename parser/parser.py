import nltk
import sys
import string
from nltk.tokenize import word_tokenize

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
D -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP | NP VP Conj NP VP

AP -> Adj | Adj AP
NP -> N | D NP | AP NP | N PP
PP -> P NP
VP -> V | V NP | V NP PP | V PP | Adv VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = word_tokenize(sentence)
    print (tokens)
    tokens = list(filter(lambda token: token not in string.punctuation, tokens))
    lowercase_tokens = [x.lower() for x in tokens]
    return lowercase_tokens

def np_chunk(tree):
    result = []
    subtrees = tree.subtrees()
    if subtrees:
        for subtree in subtrees:
            if subtree.label() == 'NP':
                if subtree.height() == 3:
                    result.append(subtree)
                else:
                    if subtree == tree:
                        continue
                    np_chunk(subtree)
                    # print (f'Sub: {sub}')
                    # print (sub.height())
                # if subtree == tree:
                #     continue
                # np_chunk(subtree)
    # print (tree)
    return result


if __name__ == "__main__":
    main()
