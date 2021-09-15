import _PyPacwar
import numpy
import random


# Example Python module in C for Pacwar
def main():

    ones = [1] * 50
    var_ones = [1] * 50
    threes = [3] * 50
    c1 = 100
    c2 = 100
    #rounds = 500
    while(c1 > 0 or c3 > 0 or rounds_1 > 80 or rounds_2 > 80):
        var_ones = [1] * 50
        index_modified=[]
        for i in range(0, 30):
            index_modified.append(random.randint(0, 49))
        for i in index_modified:
            var_ones[i] = random.randint(0, 3)
        (rounds_1, c1, c2) = _PyPacwar.battle(ones, var_ones)
        (rounds_2, c3, c2) = _PyPacwar.battle(threes, var_ones)
        print(c1, rounds_1, c3, rounds_2)
    print("Found!\n=> ", var_ones)
    print(f"String: {''.join(map(str, var_ones))}")
    print("Example Python module in C for Pacwar")
    print("all ones versus var all ones and all threes...")
    #(rounds, c1, c2) = _PyPacwar.battle(ones, threes)
    print("Number of rounds:", rounds_1)
    print("Ones PAC-mites remaining:", c1)
    #print("Var PAC-mites remaining:", c2)
    print("Number of rounds:", rounds_2)
    print("Threes PAC-mites remaining:", c3)


if __name__ == "__main__":
    main()
