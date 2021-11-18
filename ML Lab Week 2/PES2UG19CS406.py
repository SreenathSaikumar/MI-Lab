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
    from heapq import heapify,heappush,heappop
    fron=[]
    heapify(fron)
    heappush(fron,(0,start_point))
    came_from={start_point:None}
    cost_so_far={start_point:0}
    path=[]
    while(len(fron)):
        curr=heappop(fron)
        if curr[1] in goals:
            current=curr[1]
            while current is not None:
                path.append(current)
                current=came_from[current]
            path=path[::-1]
            break
        for i in range(1,len(cost)):
            if(cost[curr[1]][i]>0):
                new_cost=cost_so_far[curr[1]]+cost[curr[1]][i]
                if i not in cost_so_far or new_cost<cost_so_far[i]:
                    cost_so_far[i]=new_cost
                    prio=new_cost+heuristic[i]
                    heappush(fron,(prio,i))
                    came_from[i]=curr[1]
    return path
    # TODO



def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    visited=set()
    stack=[]
    path=[]
    stack.append(start_point)
    while(len(stack)):
        v=stack.pop()
        if v not in visited:
            visited.add(v)
            path.append(v)
            if v in goals:
                return path
            for i in range((len(cost)-1),0,-1):
                if(cost[v][i]>0):
                    stack.append(i)

    # TODO
    return []
