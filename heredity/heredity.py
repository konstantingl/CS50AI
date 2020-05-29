import csv
import itertools
from itertools import chain
import sys
import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    print (f'One gene:{one_gene}, Two gene:{two_genes}, Have:{have_trait}')
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probs = []

    for i in people.values():
        if not i['mother']:
            #Calculate 0 genes no trait with no parents
            if i['name'] not in chain(one_gene,two_genes,have_trait):
                gene_amount = 0
                trait = False

            #Calculate 0 genes have_trait with no parents
            if i['name'] not in one_gene and i['name'] not in two_genes and i['name'] in have_trait:
                gene_amount = 0
                trait = True

            #Calculate 1 gene no trait with no parents
            elif i['name'] in one_gene and i['name'] not in have_trait:
                gene_amount = 1
                trait = False

            #Calculate 1 gene have_trait with no parents
            elif i['name'] in one_gene and i['name'] in have_trait:
                gene_amount = 1
                trait = True

            #Calculate 2 genes no trait with no parents
            elif i['name'] in two_genes and i['name'] not in have_trait:
                gene_amount = 2
                trait = False

            #Calculate 2 genes have_trait with no parents
            elif i['name'] in two_genes and i['name'] in have_trait:
                gene_amount = 2
                trait = True

            # probs[i['name']] = {'gene_amount': gene_amount, 'gene_prob':PROBS['gene_amount'][gene_amount], 'trait_prob': PROBS['trait'][gene_amount][trait]}}
            prob = PROBS['gene'][gene_amount] * PROBS['trait'][gene_amount][trait]
            probs.append(prob)



        if i['mother']:

            # 0 genes
            if i['name'] not in chain(one_gene,two_genes):
                if i['mother'] not in chain(one_gene,two_genes) and i['father'] not in chain(one_gene,two_genes):
                    prob_gene = 0.99*0.99
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in one_gene:
                    prob_gene = 0.99*0.5
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in one_gene:
                    prob_gene = 0.99*0.5
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in two_genes:
                    prob_gene = 0.99*0.01
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in two_genes:
                    prob_gene = 0.99*0.01
                elif i['mother'] in one_gene and i['father'] in one_gene:
                    prob_gene = 0.5*0.5
                elif i['mother'] in one_gene and i['father'] in two_genes:
                    prob_gene = 0.5*0.01
                elif i['father'] in one_gene and i['mother'] in two_genes:
                    prob_gene = 0.5*0.01
                elif i['father'] in two_genes and i['mother'] in two_genes:
                    prob_gene = 0.01*0.01

                if i['name'] in have_trait:
                    prob = prob_gene * PROBS['trait'][0][True]
                else:
                    prob = prob_gene * PROBS['trait'][0][False]
                probs.append(prob)

            # 1 gene
            if i['name'] in one_gene:
                if i['mother'] not in chain(one_gene,two_genes) and i['father'] not in chain(one_gene,two_genes):
                    prob_gene = 0.01*0.99 + 0.01*0.99
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in one_gene:
                    prob_gene = 0.5
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in one_gene:
                    prob_gene = 0.5
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in two_genes:
                    prob_gene = 0.01 * 0.01 + 0.99*0.99
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in two_genes:
                    prob_gene = 0.01 * 0.01 + 0.99*0.99
                elif i['mother'] in one_gene and i['father'] in one_gene:
                    prob_gene = 0.5
                elif i['mother'] in one_gene and i['father'] in two_genes:
                    prob_gene = 0.5
                elif i['father'] in one_gene and i['mother'] in two_genes:
                    prob_gene = 0.5
                elif i['father'] in two_genes and i['mother'] in two_genes:
                    prob_gene = 0.01*0.99 + 0.99*0.01

                if i['name'] in have_trait:
                    prob = prob_gene * PROBS['trait'][1][True]
                else:
                    prob = prob_gene * PROBS['trait'][1][False]
                probs.append(prob)

            # 2 gene
            if i['name'] in two_genes:
                if i['mother'] not in chain(one_gene,two_genes) and i['father'] not in chain(one_gene,two_genes):
                    prob_gene = 0.01*0.01
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in one_gene:
                    prob_gene = 0.01*0.5
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in one_gene:
                    prob_gene = 0.5*0.01
                elif i['mother'] not in chain(one_gene,two_genes) and i['father'] in two_genes:
                    prob_gene = 0.01 * 0.99
                elif i['father'] not in chain(one_gene,two_genes) and i['mother'] in two_genes:
                    prob_gene = 0.01 * 0.99
                elif i['mother'] in one_gene and i['father'] in one_gene:
                    prob_gene = 0.5*0.5
                elif i['mother'] in one_gene and i['father'] in two_genes:
                    prob_gene = 0.5*0.99
                elif i['father'] in one_gene and i['mother'] in two_genes:
                    prob_gene = 0.5*0.99
                elif i['father'] in two_genes and i['mother'] in two_genes:
                    prob_gene = 0.99*0.99

                if i['name'] in have_trait:
                    prob = prob_gene * PROBS['trait'][2][True]
                else:
                    prob = prob_gene * PROBS['trait'][2][False]
                probs.append(prob)

    return (np.prod(probs))

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person not in chain(one_gene,two_genes):
            probabilities[person]['gene'][0] += p

        elif person in one_gene:
            probabilities[person]['gene'][1] += p

        elif person in two_genes:
            probabilities[person]['gene'][2] += p

        if person in have_trait:
            probabilities[person]['trait'][True] += p

        elif person not in have_trait:
            probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        norm_gene = 1/(probabilities[person]['gene'][0] + probabilities[person]['gene'][1] + probabilities[person]['gene'][2])
        norm_trait = 1/(probabilities[person]['trait'][True] + probabilities[person]['trait'][False])

        for i in range(3):
            probabilities[person]['gene'][i] = probabilities[person]['gene'][i] * norm_gene

        probabilities[person]['trait'][True] = probabilities[person]['trait'][True] * norm_trait
        probabilities[person]['trait'][False] = probabilities[person]['trait'][False] * norm_trait

if __name__ == "__main__":
    main()
