# Authors: Thomas Levitt and Dawson Merkle
# Project 2: Traveling Salesman Problem using Genetic Algorithm
# CAP4630
# 6/11/2023


# STEPS TO THE ALGORITHM

# 1. Encode the Solution Space
# 2. Set Algorithm Parameters
# 3. Create Initial Population
# 4. Measure Fitness of Individuals
# 5. Select Parents
# 6. Reproduce Offspring
# 7. Populate Next Generation
# 8. Measure Fitness of Individuals 
# 9. Repeat 5-9 Until Desired Fitness or Interations is Reached




# THINGS TO DO:

# Rewrite crossover function
# Plot routes (global/generational)
# Implement long city list and randomize city selection
# Write other selection algorithms
# TEST!!!


# DONE:

# Make more children every generation
# Determine if we want to kill off old population members
#     IF KILL OFF
#         Random homicide?
#         Weakest link homicide?
#     ELSE
#         Keep adding children to population
# Think and change mutate function


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
        return (self.x, self.y)
    
    def __repr__(self):
        '''returns string representation of city'''

        return "({name}: x:{x}, y:{y})".format(name = self.name, 
                                         x = self.x, y = self.y)
    

class Fitness():
    '''fitness class'''

    def __init__(self, route):    
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    

def calculate_fitness(population, BEST_ROUTE_LENGTH):
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
        elif totalRouteLength < BEST_ROUTE_LENGTH:
            BEST_ROUTE_LENGTH = totalRouteLength


        routeLengths.append(totalRouteLength)
    return routeLengths, BEST_ROUTE_LENGTH



def set_probability(routeLengths):
    '''sets probability for the whole population'''

    probability = []
    for route in routeLengths:
        probability.append(1 / route)

    normProbabilities = [float(i)/sum(probability) for i in probability]

    return normProbabilities


# GOT RID OF NUMBER OF PARENTS ARGUMENT AND ROUTELENGTHS
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

    index = 0

    for slice in slices:
        if randomNumber < slice:
            return population[index]
        index = index + 1


                        
def crossover(parent1, parent2):
    '''Splices parent1 at random int and appends from parent2'''

    splice1 = random.randrange(len(parent1))
    splice2 = random.randrange(len(parent1))

    startIND = min(splice1, splice2)
    endIND = max(splice1, splice2)


    child = parent1[startIND:endIND]

    for x in parent2:
        if x not in child:
            child.append(x)

    return child



    

# def crossover(parent1, parent2):
    
#     childP1 = []
    
#     geneA = int(random.random() * len(parent1))
#     geneB = int(random.random() * len(parent1))
    
#     startGene = min(geneA, geneB)
#     endGene = max(geneA, geneB)

#     for i in range(startGene, endGene):
#         childP1.append(parent1[i])
        
#     childP2 = [item for item in parent2 if item not in childP1]

#     child = childP1 + childP2
#     return child


def mutate(MUTATION_RATE, N_SIZE, child):

    randomNum = random.uniform(0, 1)

    randomIndex1 = random.randint(0, N_SIZE - 1)
    randomIndex2 = random.randint(0, N_SIZE - 1)

    if randomNum < MUTATION_RATE:

        child[randomIndex1], child[randomIndex2] = child[randomIndex2], child[randomIndex1]

    return child


def mergepopulation(population, children, routeLengths, KILL_RATE):
    '''removes members of the population then merges with the children'''

    for death in range(int(KILL_RATE * len(population))):
        worstRoute = 0
        for route in range(len(routeLengths)):
            if routeLengths[route] > worstRoute:
                worstRoute = routeLengths[route]
                worstIndex = route
            else:
                continue
        del population[worstIndex]
        del routeLengths[worstIndex]

        
    for child in children:
        population.append(child)

    return population


def create_population(N_SIZE, POP_SIZE):
    '''creates the cities and population'''

    population = []
    cityList = []

    # 25 cities
    cityName = ['Boca Raton', 'Bradenton', 'Clearwater', 'Cocoa Beach', 
                'Coral Gables', 'Daytona Beach', 'Deerfield Beach', 
                'Delray Beach', 'Fort Lauderdale', 'Fort Myers', 'Gainesville', 
                'Hollywood', 'Jacksonville', 'Key West', 'Lakeland', 
                'Melbourne', 'Miami', 'Naples', 'Orlando', 'Pensacola',
                'Pompano Beach', 'Sarasota', 'Tallahassee', 'Tampa', 
                'West Palm Beach']
    
    # 58 cities
    # cityName = ['Apalachicola', 'Bartow','Belle Glade', 'Boca Raton', 
    #             'Bradenton', 'Cape Coral', 'Clearwater', 'Cocoa Beach', 
    #             'Cocoa-Rockledge', 'Coral Gables', 'Daytona Beach', 'De Land',
    #             'Deerfield Beach', 'Delray Beach', 'Fernandina Beach',
    #             'Fort Lauderdale', 'Fort Myers', 'Fort Pierce', 
    #             'Fort Walton Beach', 'Gainesville', 'Hallandale Beach', 
    #             'Hialeah', 'Hollywood', 'Homestead', 'Jacksonville', 
    #             'Key West', 'Lake City', 'Lake Wales', 'Lakeland', 'Largo', 
    #             'Melbourne', 'Miami', 'Miami Beach', 'Naples', 
    #             'New Smyrna Beach', 'Ocala', 'Orlando', 'Ormond Beach', 
    #             'Palatka', 'Palm Bay', 'Palm Beach', 'Panama City', 
    #             'Pensacola', 'Pompano Beach', 'Saint Augustine', 
    #             'Saint Petersburg', 'Sanford', 'Sarasota', 'Sebring', 
    #             'Tallahassee', 'Tampa', 'Tarpon Springs', 'Titusville', 
    #             'Venice', 'West Palm Beach', 'White Springs', 'Winter Haven', 
    #             'Winter Park']


    
    for city in range(0, N_SIZE):
        cityList.append(City(name=cityName[city], 
                            x=int(random.random() * 200), 
                             y=int(random.random() * 200)))
        

    # Outputting the starting distance with the alphabetical city list
    # print('\n')
    # print('This is the alphabetical route distance: ')
    # alphRouteDist, BEST_ROUTE_LENGTH = calculate_fitness(list(cityList), 
    #                                                     BEST_ROUTE_LENGTH)
    # print(alphRouteDist)
    
    # COULD HAVE INITIAL DISTANCE USING THE ALPHABETICAL CITY LIST 
    # TO SEE IF GA IS WORKING


    for chromosome in range(POP_SIZE):
        # random.seed(chromosome)
        random.shuffle(cityList)
        population.append(list(cityList))

    # print(population)
    return population





# Y AXIS = DISTANCE IN UNITS
# X AXIS = GENERATIONS
# PLOT POINTS = BEST_ROUTE_LIST (its a list btw)

# def plot_distances()





# def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     progress = []
#     progress.append(1 / rankRoutes(pop)[0][1])

#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#         progress.append(1 / rankRoutes(pop)[0][1])

#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.show()



# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)



def main():
    '''main function'''


    N_SIZE = 25
    POP_SIZE = 100
    BEST_ROUTE_LENGTH = 0
    BEST_ROUTE_LIST = []
    MUTATION_RATE = 0.1
    KILL_RATE = 0.2
    NUM_CHILDREN = 20
    
    
    NUM_GENERATIONS = 500
    
    # random.seed(1)

    population = create_population(N_SIZE, POP_SIZE)


    # random.seed()


    for generations in range(NUM_GENERATIONS):

    
        routeLengths, BEST_ROUTE_LENGTH = calculate_fitness(population, 
                                                        BEST_ROUTE_LENGTH)

        normProbabilities = set_probability(routeLengths)


        parent1 = proportional_roulette_selection(population, normProbabilities)
        parent2 = proportional_roulette_selection(population, normProbabilities)


        children = []

        for x in range(NUM_CHILDREN):
            child = crossover(parent1, parent2)
            child = mutate(MUTATION_RATE, N_SIZE, child)
            children.append(child)

        population = mergepopulation(population, children, routeLengths, 
                                    KILL_RATE)
            
        BEST_ROUTE_LIST.append(BEST_ROUTE_LENGTH)
    

    print('\n')
    print('Best route starting point: ')
    print(BEST_ROUTE_LIST[0])

    print('\n')
    print('Best route length found within the generations: ')
    print(BEST_ROUTE_LENGTH)
    print('\n')

    # print('\n')
    # print('Best route list: ')
    # print(BEST_ROUTE_LIST)


    print('\n')
    print('This is the final length of the population: ')
    print(len(population))
    print('\n')



if __name__ == '__main__':
    main()





# def run_ga():
#     best_global_fitness = 0
#     global_population = generate_initial_population(INITIAL_POPULATION_SIZE)
#     for generation in range(NUMBER_OF_GENERATIONS):
#         current_best_fitness = calculate_population_fitness(global_population, KNAPSACK_WEIGHT_CAPACITY)
#         if current_best_fitness > best_global_fitness:
#             best_global_fitness = current_best_fitness
#         the_chosen = roulette_wheel_selection(global_population, 100)
#         the_children = reproduce_children(the_chosen)
#         the_children = mutate_children(the_children, MUTATION_RATE)
#         global_population = merge_population_and_children(global_population, the_children)
#         # print(global_population)