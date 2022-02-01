graph={
    1: [2,3,4],
    2: [5],
    3: [5],
    4: [5],
    5: [6,7],
    6:[],
    7:[3]
    }

def recursive_dfs(v, discovered=[]):
    discovered.append(v)
    for w in graph[v]:
        if w not in discovered:
            discovered= recursive_dfs(w, discovered)
    return discovered

def iterative_dfs(start_v):
    discovered=[]
    stack=[start_v]
    while stack:
        v=stack.pop()
        if v not in discovered:
            discovered.append(v)
            for w in graph[v]:
                stack.append(w)
    return discovered

def iterative_bfs(start_v):
    discovered=[start_v]
    queue=[start_v]
    while queue:
        v=queue.pop(0)
        for w in graph[v]:
            if w not in discovered:
                discovered.append(w)
                queue.append(w)

    return discovered


if __name__=='__main__':
    
    print(f"Recursive DFS: {recursive_dfs(1)}")
    print(f"Iterative DFS: {iterative_dfs(1)}")
    print(f"Iterative BFS: {iterative_bfs(1)}")
    
