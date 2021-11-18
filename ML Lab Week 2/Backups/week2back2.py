"""
You can create any other helper funtions.
Do not modify the given functions
"""


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    visited=[0]*len(cost)
    visited[start_point]=1
    from heapq import heapify,heappush,heappop
    heap=[]
    heapify(heap)
    heappush(heap,(0,start_point))
    path = []
    path.append(start_point)
    while(len(heap)):
        print('here')
        s=heappop(heap)
        s=s[1]
        print(s)
        if(visited[s]==0):
            visited[s]=1
            path.append(s)
            if s in goals:
                break
        for i in range(1,len(cost)):
            if(cost[s][i]!=-1 and cost[s][i]!=0):
                heappush(heap,(cost[s][i],i))
    # TODO
    #print(path)
    return path
