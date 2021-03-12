import sys
from car import Car

# Episodes, 1-100, each episode is a thousand steps.
episodes = 20
# Number of cars, 2-50.
nb_cars = 50


if len(sys.argv) == 3 or len(sys.argv) == 5:
    if (sys.argv[1] == "-nc") and sys.argv[2].isdigit() and 2 <= int(sys.argv[2]) <= 50:
        nb_cars = int(sys.argv[2])
    if (sys.argv[1] == "-e") and sys.argv[2].isdigit() and 1 <= int(sys.argv[2]) <= 100:
        episodes = int(sys.argv[2])
    if len(sys.argv) == 5:
        if (sys.argv[3] == "-nc")  and sys.argv[4].isdigit() and 3 <= int(sys.argv[4]) <= 50:
            nb_cars = int(sys.argv[4])
        if (sys.argv[3] == "-e") and sys.argv[4].isdigit() and 1 <= int(sys.argv[4]) <= 100:
            episodes= int(sys.argv[4])

# main entry point
if __name__ == "__main__":

    sys.argv = [sys.argv[0]]

    print(f"The number of cars is: {nb_cars}")
    print(f"The number of steps is: {episodes}000")

    car = Car('network/road.net.xml',nb_cars)
    car.Q_learning(episodes)

    car.simulation_close()

