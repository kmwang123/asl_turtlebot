# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:29:09 2020

@author: steve
"""
import numpy as np
from tspy import TSP
from tspy.solvers import TwoOpt_solver
import itertools
from P1_astar import *

#D = np.array([[1., 2., 3.],
#              [4., 5., 6.],
#              [7., 8., 9.]])
#
#tsp = TSP()
#
## Using the distance matrix
#tsp.read_mat(D)
#
#two_opt = TwoOpt_solver(initial_tour='NN', iter_num=100)
#
#two_opt_tour = tsp.get_approx_solution(two_opt)
#
#best_tour = tsp.get_best_solution()

def solve_tsp(robot_loc, pickup_locs, astar):

    # build distance matrix
    nodes = [robot_loc] + pickup_locs  # concatenate
    n_nodes = len(nodes)

    D = np.zeros([n_nodes, n_nodes])
    paths =  [ [ None for i in range(n_nodes) ] for j in range(n_nodes) ]
    for i, node_1 in enumerate(nodes):
        for j, node_2 in enumerate(nodes):
            if i == j:
                D[i,j] = np.inf
            if(j > i):
                astar.reset(node_1, node_2)
                path = astar.solve()
                if not path:
                    D[i,j] = np.inf
                else:
                    D[i,j] = astar.cost_to_arrive[astar.x_goal]
                    D[j,i] = D[i,j]
                    paths[i][j] = paths[j][i] = path

    # let's do some shenanigans to make it so we dont care about returning to starting point
    #which the traveling salesman prob cares about. Just zero out the dist farthest from robot!
    D_to_robot = D[:,0]
    D_to_robot[0] = 0
    max_idx = np.argmax(D_to_robot)
    print(max_idx)
    D[max_idx, 0] = 0.  # set the return path distance for farthest node from robot to 0
#    D[0, max_idx] = 0.

    tsp = TSP()
    tsp.read_mat(D)
    print(D)
    two_opt = TwoOpt_solver(initial_tour='NN', iter_num=100)
    two_opt_tour = tsp.get_approx_solution(two_opt)
    best_tour = tsp.get_best_solution()

    #eliminate the last node as it's identical to first
    best_tour = best_tour[:-1]
    # the robot's node is represented by index 0 in distance table
    start_idx = best_tour.index(0)  # find where it starts
    # rotate the best tour so first element is always zero. Will not affect path length, it's a loop
    best_tour = best_tour[start_idx:] + best_tour[:start_idx]
    print(best_tour)
    best_tour_paths = []
    for i in range(n_nodes - 1):
        path = paths[best_tour[i]][best_tour[i+1]]
        best_tour_paths.append(path)

    return best_tour_paths
#    return best_tour




