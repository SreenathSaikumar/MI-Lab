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
    open_set=set()
    open_set.add(start_point)
    closed_set=set()
    g={}
    parents={}
    g[start_point]=0
    parents[start_point]=start_point
    while(len(open_set)):
        n=None
        for v in open_set:
            if n==None or g[v]+heuristic[v]<g[n]+heuristic[n]:
                n=v
        if n in goals or no_path(cost[n]):
            pass
        else:
            for i in range(len(cost)-1,0,-1):
                if(cost[n][i]>0):
                    if i not in open_set and i not in closed_set:
                        open_set.add(i)
                        parents[i]=n
                        g[i]=g[n]+cost[n][i]
        if n==None:
            return []
        if n in goals:
            path=[]
            while parents[n]!=n:
                path.append(n)
                n=parents[n]
                path.append(start_point)
                path.reverse()
                print(path)
                return path
        open_set.remove(n)
        closed_set.add(n)
    # TODO
    return []

def no_path(cost):
    flag=True
    for i in cost:
        if i>0:
            flag=False
    return flag



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
                print(path)
                return path
            for i in range((len(cost)-1),0,-1):
                if(cost[v][i]>0):
                    stack.append(i)

    # TODO
    print(path)
    return []
