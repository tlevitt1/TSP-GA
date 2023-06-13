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
    
    bestFitness = 0
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
        

    print('\n')
    print('THESE ARE THE ROUTE LENGTHS')
    print(routeLengths)
    print('\n')

    # raw = routeLengths
    # print(raw)

    # normRouteLengths = [float(i)/sum(raw) for i in raw]
    print('\n')
    # print(normRouteLengths)

    probability = []

    for route in routeLengths:
        probability.append(1 / route)

    print(probability)
    print('\n')



    raw = probability

    normProbabilities = [float(i)/sum(raw) for i in raw]
    print('\n')
    print(normProbabilities)
    print('\n')

    parents = []
    slices = []
    total = 0
    number_of_selections = 2


    # for i in range(0, number_of_selections):
    #     slices.append([i, total, total + normProbabilities[i]])
    #     total += normProbabilities[i]
    # spin = random.uniform(0,1)
    # result = [slice for slice in slices if slice[1] < spin <= slice[2]]
    # print('\n')
    # print(result)
    # print('\n')


    print('\n')
    randomNumber = random.uniform(0,1)
    print(randomNumber)
    print('\n')

    slice1 = normProbabilities[0]
    slice2 = slice1 + normProbabilities[1]
    slice3 = slice2 + normProbabilities[2]
    slice4 = slice3 + normProbabilities[3]
    slice5 = slice4 + normProbabilities[4]
    slice6 = slice5 + normProbabilities[5]
    slice7 = slice6 + normProbabilities[6]
    slice8 = slice7 + normProbabilities[7]
    slice9 = slice8 + normProbabilities[8]
    # slice10 = slice9 + normProbabilities[9]




    if randomNumber < slice1:
        print('Route 1 has been chosen as a parent')
        parents.append(normProbabilities[0])
    elif randomNumber < slice2:
        print('Route 2 has been chosen as a parent')
        parents.append(normProbabilities[1])
    elif randomNumber < slice3:
        print('Route 3 has been chosen as a parent')
        parents.append(normProbabilities[2])
    elif randomNumber < slice4:
        print('Route 4 has been chosen as a parent')
        parents.append(normProbabilities[3])
    elif randomNumber < slice5:
        print('Route 5 has been chosen as a parent')
        parents.append(normProbabilities[4])
    elif randomNumber < slice6:
        print('Route 6 has been chosen as a parent')
        parents.append(normProbabilities[5])
    elif randomNumber < slice7:
        print('Route 7 has been chosen as a parent')
        parents.append(normProbabilities[6])
    elif randomNumber < slice8:
        print('Route 8 has been chosen as a parent')
        parents.append(normProbabilities[7])
    elif randomNumber < slice9:
        print('Route 9 has been chosen as a parent')
        parents.append(normProbabilities[8])
    else:
        print('Route 10 has been chosen as a parent')
        parents.append(normProbabilities[9])

    print('\n')
    print(parents)



    # for route in routeLengths:
    #     probability.append(route / sum(routeLengths))

    # print(probability)


    
# def set_probabilities_of_population(population):
#     total_fitness = sum of fitness 
#     for individual in population:
#         probability_of_selection of individual = fitness / total_fitness


# def roulette_wheel_selection(population, normRouteLengths):
#     possibleProbabilities = set_probabilities_of_population(population)
#     slices = []
#     total = 0
#     for i in range(0, len(normRouteLengths)):
#         slices.append(i, total, total + possible_probabilities[i])
#         total += possibleProbabilities[i]
#     spin = random.random(0, 1)
#     result = [slice for slice in slices if slice[1] < spin <= slice[2]]
#     return result



    # ###############################################
    # import numpy.random as npr
    # def selectOne(self, population):
    # max = sum([c.fitness for c in population])
    # selection_probs = [c.fitness/max for c in population]
    # return population[npr.choice(len(population), p=selection_probs)]
    # ################################################
    # def selectOne(self, population):
    # max = sum([c.fitness for c in population])
    # pick = random.uniform(0, max)
    # current = 0
    # for chromosome in population:
    #     current += chromosome.fitness
    #     if current > pick:
    #         return chromosome







# def calculate_population_fitness(population, maximum_weight):
#     best_fitness = 0
#     for individual in population:
#         individual_fitness = calculate_individual_fitness(individual[INDIVIDUAL_CHROMOSOME_INDEX], maximum_weight)
#         individual[INDIVIDUAL_FITNESS_INDEX] = individual_fitness
#         if individual_fitness > best_fitness:
#             best_fitness = individual_fitness
#         # if individual_fitness == -1:
#         #     population.remove(individual)
#     return best_fitness


# def calculate_individual_fitness(individual, maximum_weight):
#     total_individual_weight = 0
#     total_individual_value = 0
#     for gene_index in range(len(individual)):
#         gene_switch = individual[gene_index]
#         if gene_switch == '1':
#             total_individual_weight += knapsack_items[gene_index][KNAPSACK_ITEM_WEIGHT_INDEX]
#             total_individual_value += knapsack_items[gene_index][KNAPSACK_ITEM_VALUE_INDEX]
#     if total_individual_weight > maximum_weight:
#         return -1
#     return total_individual_value




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

    # Debugging
    print('\n\n')
    print(population[0])
    print('\n')
    # print(population[1])
    # print('\n')
    # print(population[2])
    # print('\n')
    # print(len(population))
    # print('\n')

    return population




def main():
    '''main function'''

    N_SIZE = 25
    POP_SIZE = 10

    population = create_population(N_SIZE, POP_SIZE)
    
    calculate_fitness(population)











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