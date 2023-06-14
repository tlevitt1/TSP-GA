# Authors: Thomas Levitt and Dawson Merkle
# Project 2: Traveling Salseman Problem using Genetic Algorithm
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




import random
import numpy as np


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
    

def calculate_fitness(population):
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

        routeLengths.append(totalRouteLength)
    return routeLengths



def set_probability(routeLengths):
    '''sets probability for the whole population'''

    probability = []
    for route in routeLengths:
        probability.append(1 / route)
        
    # print(probability)

    normProbabilities = [float(i)/sum(probability) for i in probability]
    # print(normProbabilities)

    return normProbabilities


def proportional_roulette_selection(NUM_PARENTS, routeLengths, population, 
                                    normProbabilities):
    '''Selecting parents using proportional roulette selection'''
    
    normProbabilities = set_probability(routeLengths)


    slices = []

    for slice in range(len(normProbabilities)):
        sliceCounter = 0
        if slice == 0:
            slices.append(normProbabilities[0])
        else:

            slices.append(slices[sliceCounter-1] + normProbabilities[slice])

    # print(slices)

    parents = []
    prevSlice = None

    while len(parents) != NUM_PARENTS:

        randomNumber = random.uniform(0,1)
        #print('ran', randomNumber)
        for index, slice in enumerate(slices):

            if prevSlice == index:
                continue
            elif randomNumber < slice:

                #print('sli', slice)
                                #ROUTE 
                parents.append(population[index])
                print(parents)
                prevSlice = index
                break
            else:
                continue

    return parents

    # while len(parents) != NUM_PARENTS:

    #     randomNumber = random.uniform(0,1)
    #     #print('ran', randomNumber)
    #     sliceCounter = 0
    #     for slice in slices:

    #         if prevSlice == slice:
    #             break

    #         if randomNumber < slice:
            
    #             #print('sli', slice)
    #                             #ROUTE 
    #             parents.append(population[sliceCounter])
    #             prevSlice = slice
    #             break
                        
    #         sliceCounter += 1

    # return parents


def create_population(N_SIZE, POP_SIZE):
    '''creates the cities and population'''

    population = []
    cityList = []

    #25 cities
    cityName = ['Boca Raton', 'Bradenton', 'Clearwater', 'Cocoa Beach', 
                'Coral Gables', 'Daytona Beach', 'Deerfield Beach', 
                'Delray Beach', 'Fort Lauderdale', 'Fort Myers', 'Gainesville', 
                'Hollywood', 'Jacksonville', 'Key West', 'Lakeland', 
                'Melbourne', 'Miami', 'Naples', 'Orlando', 'Pensacola',
                'Pompano Beach', 'Sarasota', 'Tallahassee', 'Tampa', 
                'West Palm Beach']
    
    for city in range(0, N_SIZE):
        cityList.append(City(name=cityName[city], 
                            x=int(random.random() * 200), 
                             y=int(random.random() * 200)))
        
    
    # COULD HAVE INITIAL DISTANCE USING THE ALPHABETICAL CITY LIST 
    # TO SEE IF GA IS WORKING


    for chromosome in range(POP_SIZE):
        # random.seed(chromosome)
        random.shuffle(cityList)
        population.append(list(cityList))

    # print(population)
    return population




def main():
    '''main function'''

    N_SIZE = 25
    POP_SIZE = 10

    # CHANG
    NUM_PARENTS = 2

    population = create_population(N_SIZE, POP_SIZE)
    
    routeLengths = calculate_fitness(population)

    normProbabilities = set_probability(routeLengths)

    parents = proportional_roulette_selection(NUM_PARENTS, routeLengths, population, 
                                    normProbabilities)

    print('\n')
    print(parents[0])
    print('\n')
    print(parents[1])
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