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
        if self.distance ==0:
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
    
    


def main():
    '''main function'''

    # random.seed(2)


    N_SIZE = 25

    population = []
    popSize = 10

    
    cityName = ['Boca Raton', 'Bradenton', 'Clearwater', 'Cocoa Beach', 
                'Coral Gables', 'Daytona Beach', 'Deerfield Beach', 
                'Delray Beach', 'Fort Lauderdale', 'Fort Myers', 'Gainesville', 
                'Hollywood', 'Jacksonville', 'Key West', 'Lakeland', 
                'Melbourne', 'Miami', 'Naples', 'Orlando', 'Pensacola',
                'Pompano Beach', 'Sarasota', 'Tallahassee', 'Tampa', 
                'West Palm Beach']
    
    # for chromo in range(0, 10):
    #     cityList = []

    #     for city in range(0, N_SIZE):
    #         cityList.append(City(name=cityName[city], 
    #                             x=int(random.random() * 200), 
    #                             y=int(random.random() * 200)))
            
    #     random.shuffle(cityList)
    #     population.append(cityList)


    cityList = []

    for city in range(0, N_SIZE):
        cityList.append(City(name=cityName[city], 
                            x=int(random.random() * 200), 
                             y=int(random.random() * 200)))
        

    for x in range(popSize):
        # random.seed(x)
        random.shuffle(cityList)
        population.append(list(cityList))



    print('\n\n')

    print(population[0])
    print('\n')
    print(population[1])
    print('\n')
    print(population[2])
    print('\n')
    print(len(population))
    print('\n')

    
if __name__ == '__main__':
    main()