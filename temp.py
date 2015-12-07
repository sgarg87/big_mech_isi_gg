import numpy.random as rnd


a = 0

def a1(seed):
    a = seed
    rnd.seed(a)
    a_print()


def a_print():
    print rnd.random(10)


if __name__ == '__main__':
    a1(1)
    a1(1)
    a1(2)
    a1(3)
    a1(4)

