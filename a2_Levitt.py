# Authors: Thomas Levitt and Dawson Merkle
# Project 2: Traveling Salesman Problem using Genetic Algorithm
# CAP4630
# 6/11/2023


import random
import numpy as np
import matplotlib.pyplot as plt


class City():
    '''city class'''

    def __init__(self, name, x, y):
        '''constructor'''
        self.name = name
        self.x = x
        self.y = y
    
    def get_distance(self, other):
        '''gets distance from one city from another'''
        xDis = abs(self.x - other.x)
        yDis = abs(self.y - other.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def get_coords(self):
        '''returns tuple of x and y'''
        return self.x, self.y
    
    def __repr__(self):
        '''returns string representation of city'''

        return "({name}: x:{x}, y:{y})".format(name = self.name, 
                                         x = self.x, y = self.y)


def plot_generations_and_route(BEST_ROUTE_LIST, NUM_GENERATIONS, BEST_ROUTE, 
                               BEST_ROUTE_LENGTH):
    '''Plots the best routes per generation and the best route length with
        a diagram showing where the cities are'''
    
    plt.figure(figsize=(9.5,9.5))

    # Plotting the Generations plot
    xaxis = list(range(0, NUM_GENERATIONS))

    plt.subplot(2, 1, 1)
    plt.plot(xaxis, BEST_ROUTE_LIST)
    plt.title('Best Routes Each Generation')
    plt.xlabel('Generation')
    plt.ylabel('Route Distance')
    

    # Plotting the Best Route plot
    xLis = []
    yLis = []
    for city in BEST_ROUTE:

        xPos, yPos = city.get_coords()

        xLis.append(xPos)
        yLis.append(yPos)

    xLast, yLast = BEST_ROUTE[0].get_coords()
    xLis.append(xLast)
    yLis.append(yLast)

    plt.subplot(2, 1, 2)
    plt.axis([0, 200, 0, 200])

    plt.plot(xLis, yLis, c='black', mfc='red', marker='o',
             markersize='7', label='Cities')
    plt.plot(xLis[0], yLis[0], c='black', mfc='green', marker='s', 
             markersize='9', label='Start')
    plt.plot(xLis[1], yLis[1], c='black', mfc='purple', marker='D', 
             markersize='9', label='Next')
    plt.title('Best Route with length of: ' + str(round(BEST_ROUTE_LENGTH, 3)))
    plt.legend()


    # Plot both subplots
    plt.tight_layout(pad=4.0)
    plt.show()


def mergepopulation(population, children, routeLengths, KILL_RATE):
    '''removes members of the population then merges with the children'''

    # killing members of population 
    for _ in range(int(KILL_RATE * len(population))):
        worstRoute = 0
        for route in range(len(routeLengths)):
            if routeLengths[route] > worstRoute:
                worstRoute = routeLengths[route]
                worstIndex = route
            else:
                continue
        del population[worstIndex]
        del routeLengths[worstIndex]

    # Adding children
    for child in children:
        population.append(child)

    return population


def mutate(MUTATION_RATE, N_SIZE, child):
    '''mutates the chromosome by picking two indices and swapping'''

    randomNum = random.uniform(0, 1)

    randomIndex1 = random.randint(0, N_SIZE - 1)
    randomIndex2 = random.randint(0, N_SIZE - 1)

    if randomNum < MUTATION_RATE:

        child[randomIndex1], child[randomIndex2] = child[randomIndex2], child[randomIndex1]

    return child


def crossover(parent1, parent2):
    '''Splices parent1 from random range and appends from parent2'''

    splice1 = random.randrange(len(parent1))
    splice2 = random.randrange(len(parent1))

    startIND = min(splice1, splice2)
    endIND = max(splice1, splice2)

    child = parent1[startIND:endIND]

    for city in parent2:
        if city not in child:
            child.append(city)

    return child


def proportional_roulette_selection(population, normProbabilities):
    '''Selecting parents using proportional roulette selection'''

    slices = []
    for slice in range(len(normProbabilities)):
        sliceCounter = 0
        if slice == 0:
            slices.append(normProbabilities[0])
        else:

            slices.append(slices[sliceCounter-1] + normProbabilities[slice])

    randomNumber = random.uniform(0,1)

    for index, slice in enumerate(slices):
        if randomNumber < slice:
            return population[index]


def set_probability(routeLengths):
    '''sets probability for the whole population'''

    probability = []
    for route in routeLengths:
        probability.append(1 / route)

    normProbabilities = [float(i)/sum(probability) for i in probability]

    return normProbabilities


def calculate_fitness(population, BEST_ROUTE_LENGTH, BEST_ROUTE):
    '''Calculates fitness for each route in population'''
    
    routeLengths = []
    for route in population:
        
        totalRouteLength = 0
        nextCityIndex = 1
        for city in route:

            if nextCityIndex == len(route):
                distBetweenCities = route[0].get_distance(route[nextCityIndex-1])
                totalRouteLength = totalRouteLength + distBetweenCities
                break

            distBetweenCities = city.get_distance(route[nextCityIndex])
            totalRouteLength = totalRouteLength + distBetweenCities
            nextCityIndex += 1

        # Keeping track of the global best route
        if BEST_ROUTE_LENGTH == 0:
            BEST_ROUTE_LENGTH = totalRouteLength
            BEST_ROUTE = route
        elif totalRouteLength < BEST_ROUTE_LENGTH:
            BEST_ROUTE_LENGTH = totalRouteLength
            BEST_ROUTE = route


        routeLengths.append(totalRouteLength)

    return routeLengths, BEST_ROUTE_LENGTH, BEST_ROUTE


def create_population(N_SIZE, POP_SIZE):
    '''creates the cities and population'''

    # 58 cities
    cityName = ['Apalachicola', 'Bartow','Belle Glade', 'Boca Raton', 
                'Bradenton', 'Cape Coral', 'Clearwater', 'Cocoa Beach', 
                'Cocoa-Rockledge', 'Coral Gables', 'Daytona Beach', 'De Land',
                'Deerfield Beach', 'Delray Beach', 'Fernandina Beach',
                'Fort Lauderdale', 'Fort Myers', 'Fort Pierce', 
                'Fort Walton Beach', 'Gainesville', 'Hallandale Beach', 
                'Hialeah', 'Hollywood', 'Homestead', 'Jacksonville', 
                'Key West', 'Lake City', 'Lake Wales', 'Lakeland', 'Largo', 
                'Melbourne', 'Miami', 'Miami Beach', 'Naples', 
                'New Smyrna Beach', 'Ocala', 'Orlando', 'Ormond Beach', 
                'Palatka', 'Palm Bay', 'Palm Beach', 'Panama City', 
                'Pensacola', 'Pompano Beach', 'Saint Augustine', 
                'Saint Petersburg', 'Sanford', 'Sarasota', 'Sebring', 
                'Tallahassee', 'Tampa', 'Tarpon Springs', 'Titusville', 
                'Venice', 'West Palm Beach', 'White Springs', 'Winter Haven', 
                'Winter Park']
    
    random.shuffle(cityName)
    
    cityList = []
    for city in range(0, N_SIZE):
        cityList.append(City(name=cityName[city], 
                            x=int(random.random() * 200), 
                             y=int(random.random() * 200)))
        
    # Shuffling chromosomes
    population = []
    for _ in range(POP_SIZE):
        # random.seed(chromosome)
        random.shuffle(cityList)
        population.append(list(cityList))

    # print(population)
    return population


def main():
    '''main function'''

    N_SIZE = 25
    NUM_GENERATIONS = 5000
    POP_SIZE = 100

    BEST_ROUTE_LENGTH = 0
    BEST_ROUTE_LIST = []
    BEST_ROUTE = []

    MUTATION_RATE = 0.1
    KILL_RATE = 0.2
    NUM_CHILDREN = 20
    
    
    # random.seed(1)
    population = create_population(N_SIZE, POP_SIZE)
    # random.seed()

    for generations in range(NUM_GENERATIONS):

        routeLengths, BEST_ROUTE_LENGTH, BEST_ROUTE = calculate_fitness(
                                                        population, 
                                                        BEST_ROUTE_LENGTH,
                                                        BEST_ROUTE)

        normProbabilities = set_probability(routeLengths)

        parent1 = proportional_roulette_selection(population, normProbabilities)
        parent2 = proportional_roulette_selection(population, normProbabilities)

        children = []
        for _ in range(NUM_CHILDREN):
            child = crossover(parent1, parent2)
            child = mutate(MUTATION_RATE, N_SIZE, child)
            children.append(child)

        population = mergepopulation(population, children, routeLengths, 
                                    KILL_RATE)
        
        # plots the first and subsequent generations 
        if (generations + 1) % 500 == 0 or generations == 0:
            plot_generations_and_route(BEST_ROUTE_LIST, generations, 
                                       BEST_ROUTE, BEST_ROUTE_LENGTH)
                
        BEST_ROUTE_LIST.append(BEST_ROUTE_LENGTH)
    

    print('\n')
    print('Best route starting point: ')
    print(BEST_ROUTE_LIST[0])

    # print('\n')
    # print('Best route list: ')
    # print(BEST_ROUTE_LIST)

    print('\n')
    print('Best route length found within the generations: ')
    print(BEST_ROUTE_LENGTH)
    print('\n')

    # print('\n')
    # print('This is the final length of the population: ')
    # print(len(population))

    # print('\n')
    # print('This is the best route: ')
    # print(BEST_ROUTE)

if __name__ == '__main__':
    main()
