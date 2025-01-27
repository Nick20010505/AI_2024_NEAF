import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

x = []
y = []
gene_num = 2
chromosome = 10
dis = 50
num = 2
ans_x = np.random.rand()
ans_y = np.random.rand()
population = np.random.rand(chromosome,gene_num)

iteration = 20
plt.ion()
for lter in range(iteration):
    distance=[] 
    for j in range(chromosome):
        # crossover
        if np.random.rand(1) < 0.3:                   # Randomly swap with a column
            index = int((np.random.rand(1)*10)//1)
            col = int((np.random.rand(1)*2)//1)
            
            temp = population[j, col]            
            population[j, col] = population[index, col]
            population[index,col] = temp
            
        # mutation
        if np.random.rand(1) < 0.2:
            population[j,0] = ans_x+(np.random.rand(1))*dis
            population[j,1] = ans_y+(np.random.rand(1))*dis
        
        d = math.sqrt((population[j,0]-ans_x)**2+(population[j,1]-ans_y)**2)  # Calculate the distance of each
        distance.append(d)
    s = np.argmin(distance)                           # Get the index of the shortest distance
        
    if distance[s] < dis:   
        dis = distance[s]
        best = [population[s, 0],population[s, 1]]
    
    dist = np.copy(np.argsort(distance))
    
    ## selection
    for n in range(num):
        population[dist[chromosome-1-n], 0] = population[dist[n], 0]
        population[dist[chromosome-1-n], 1] = population[dist[n], 1]
        
    plt.clf()
    plt.scatter(ans_x, ans_y, marker='*', color='r', s=100, alpha=1)
    plt.scatter(population[0:chromosome,0], population[0:chromosome,1], marker='o', color='b', s=60, alpha=0.3)
    plt.scatter(best[0], best[1], marker='+', color='g', s=100, alpha=1)
    plt.title("Iter: "+ str(lter + 1) + ", error: "+str(dis))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
    plt.pause(1.5)    
    
plt.ioff()  # Turn off interactive mode
plt.show()