import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def cell_formatter(i, row):

    if i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
        return int(row[i])
    elif i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
        return float(row[i])
    elif i==10:
        list_of_month_index = ['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
        return list_of_month_index.index(row[i])
    elif i==15:
        return 0 if row[i]=='Returning_Visitor' else 1
    else:
        return 0 if row[i]=='FALSE' else 1


def load_data(filename):

    with open("shopping.csv") as f:
        reader = csv.reader(f)
        next(reader)

        evidence = []
        lables = []
        for row in reader:
            evidence.append([cell_formatter(i, row) for i in range(17)])
            lables.append(cell_formatter(17,row))

    return (evidence,lables)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = GaussianNB()

    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity_list=[]
    specificity_list=[]
    for i in range(len(labels)):
        if labels[i] == 1:
            sensitivity_list.append(1-predictions[i])
        else:
            specificity_list.append(1-predictions[i])
    sensitivity = (len(sensitivity_list) - np.sum(sensitivity_list)) / len(sensitivity_list)
    specificity = np.sum(specificity_list) / len(specificity_list)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
