import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 100


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages_proba = dict()

    links = corpus.get(page)

    if len(links) > 0:
        for link in corpus.keys():
            if link in links:
                probability = (damping_factor / len(links)) + ((1 - damping_factor) / len(corpus))
                pages_proba[link] = probability
            else:
                probability = (1 - damping_factor) / len(corpus)
                pages_proba[link] = probability
        return pages_proba

    for link in corpus.keys():
        pages_proba[link] = 1 / len(corpus)
    return pages_proba

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = dict()
    samples = []

    first_page = random.choice(list(corpus.keys()))
    samples.append(first_page)

    for i in range(n-1):
        pages = transition_model(corpus, samples[i],damping_factor)
        selection = np.random.choice(list(pages.keys()), 1, p = list(pages.values()))
        samples.append(selection[0])

    for object in corpus.keys():
        count = 0
        for i in range(len(samples)):
            if object in samples[i]:
                count += 1
        pageranks[object] = count / len(samples)

    return pageranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = dict()
    pageranks_new = dict()
    deltas = dict()

    for page in corpus.keys():
        pageranks[page] = 1 / len(corpus)

    while True:
        for page in corpus.keys():
            keys = []
            for i,j in corpus.items():
                if page in j:
                    keys.append(i)

            proba = 0
            for key in keys:
                proba_page = pageranks[key] / len(corpus[key])
                proba += proba_page

            new_proba = ((1 - damping_factor) / len(corpus)) + damping_factor * proba

            delta = new_proba - pageranks[page]

            pageranks_new[page] = new_proba

            deltas[page] = delta
            if all(-0.001 < i < 0.001 for i in deltas.values()):
                return pageranks

        pageranks = pageranks_new.copy()





if __name__ == "__main__":
    main()
