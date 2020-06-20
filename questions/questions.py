import nltk
import sys
import os
import string
from nltk.tokenize import word_tokenize
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = dict()

    for text_name in os.listdir(directory):
        folder_path = os.path.join(directory,text_name)

        f = open(folder_path, 'r')
        contents = f.read()

        data.update({text_name: contents})

    return data

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = word_tokenize(document)

    lowercase_tokens = [x.lower() for x in tokens]

    tokens = list(filter(lambda token: token not in string.punctuation and
                         token not in nltk.corpus.stopwords.words("english"), lowercase_tokens))

    return tokens

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = []
    for text in documents:
        words.extend(documents[text])

    for word in words:
        f = sum(word in documents[name] for name in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs

def count_freq(string, word):
    count = 0
    for i in range(0, len(string)):
        if word == string[i]:
            count += 1
    return count


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    rating = {
        filename: 0
        for filename in files
    }
    for word in query:
        for file in files:
            if word in files[file]:
                rating[file] += idfs[word] * count_freq(files[file],word)

    rating_sorted = sorted(rating, key=rating.get, reverse=True)
    return rating_sorted[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rating = {
        sentence: {'idfs': 0, 'density': 0}
        for sentence in sentences
    }

    for word in query:
        for sentence in sentences:
            if word in sentences[sentence]:
                rating[sentence]['idfs'] += idfs[word]
                rating[sentence]['density'] += 1 / len(sentences[sentence])

    sorted_sentences = sorted(rating.items(), key=lambda x: (x[1]['idfs'],x[1]['density']), reverse=True)

    return [s[0] for s in sorted_sentences[:n]]

if __name__ == "__main__":
    main()
