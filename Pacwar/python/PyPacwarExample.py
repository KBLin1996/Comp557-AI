import _PyPacwar
import numpy
import random


# Example Python module in C for Pacwar
def main():

    ones = [1] * 50
    var_ones = [1] * 50
    #threes = [3] * 50
    c1 = 100
    c2 = 100
    rounds = 500
    while(c1 > 0 or rounds > 100):
        var_ones = [1] * 50
        index_modified=[]
        for i in range(0, 10):
            index_modified.append(random.randint(0, 49))
        for i in index_modified:
            var_ones[i] = random.randint(0, 3)
        (rounds, c1, c2) = _PyPacwar.battle(ones, var_ones)
        print(c1, rounds)
    print("Found!\n=> ", var_ones)
    print("Example Python module in C for Pacwar")
    print("all ones versus var all ones ...")
    #(rounds, c1, c2) = _PyPacwar.battle(ones, threes)
    print("Number of rounds:", rounds)
    print("Ones PAC-mites remaining:", c1)
    print("Var PAC-mites remaining:", c2)


if __name__ == "__main__":
    main()
