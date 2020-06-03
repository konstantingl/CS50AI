import sys
import itertools
import operator

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            for word in self.domains[v].copy():
                if len(word) != v.length:
                    self.domains[v].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        changes = 0

        if self.crossword.overlaps[(x,y)]:
            cross = list(self.crossword.overlaps[(x,y)])
            XletterNum = cross[0]
            YletterNum = cross[1]
            letters=[]
            for Yword in self.domains[y]:
                letters.append((list(Yword))[YletterNum])
            for Xword in self.domains[x].copy():
                    if all(list(Xword)[XletterNum] != letter for letter in letters):
                        if Xword in self.domains[x]:
                            self.domains[x].remove(Xword)
                        changes+=1
        if changes == 0:
            return False
        return True

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = list(itertools.combinations(self.crossword.variables, 2))
        while queue:
            arc = queue[0]
            if self.revise(arc[0], arc[1]):
                if len(self.domains[arc[0]]) == 0:
                    return False
            queue.remove(arc)
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.crossword.variables:
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        if len(assignment)>1:
            for v1 in assignment:
                for v2 in assignment:
                    if v1 != v2:
                        cross = self.crossword.overlaps[(v1,v2)]
                        if cross:
                            if v1.direction == 'ACROSS' and v2.direction == 'DOWN':
                                v1numLetter = cross[1]
                                v2numLetter = cross[0]
                            else:
                                v1numLetter = cross[0]
                                v2numLetter = cross[1]
                            if list(assignment[v1])[v1numLetter] != list(assignment[v2])[v2numLetter]:
                                return False
                        if assignment[v1] == assignment[v2]:
                            return False
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        not_assigned = []
        for var2 in self.crossword.variables:
            if var2 not in assignment:
                not_assigned.append(var2)

        values_dict = dict()
        for value in self.domains[var]:
            for var2 in not_assigned:
                eliminate=0
                if var != var2:
                    cross = self.crossword.overlaps[(var,var2)]
                    if cross:
                        if var.direction == 'ACROSS' and var2.direction == 'DOWN':
                            v1numLetter = cross[1]
                            v2numLetter = cross[0]
                        else:
                            v1numLetter = cross[0]
                            v2numLetter = cross[1]
                        for var2word in self.domains[var2]:
                            if list(value)[v1numLetter] != list(var2word)[v2numLetter]:
                                eliminate+=1
            values_dict.update({value: eliminate})
        sorted_dict = dict(sorted(values_dict.items(), key = operator.itemgetter(1)))
        return sorted_dict.keys()

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        var_dict = dict()
        for var in self.crossword.variables:
            if var not in assignment:
                domains = self.domains[var]
                var_dict.update({var:len(domains)})
        if var_dict:
            return min(var_dict, key=var_dict.get)
        return False

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        else:
            var = self.select_unassigned_variable(assignment)
            if not var:
                var = list(self.crossword.variables)[0]

            for value in self.order_domain_values(var, assignment):
                assignment.update({var: value})

                if self.consistent(assignment):
                    result = self.backtrack(assignment)
                    if result != 'failure':
                        return result
            del assignment[var]
            return 'failure'



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
