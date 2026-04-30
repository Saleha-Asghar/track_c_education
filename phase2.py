import time
from collections import deque
import heapq
import random
import math
from phase1 import df, graph  # Ensure these are imported correctly

class AIAgent:
    def __init__(self, graph):
        self.graph = graph
        self.nodes_evaluated = 0  # FIX: Initialize this here

    def perceive(self, state):
        return list(self.graph.get(state, []))

    def act(self, action):
        return action

    def goal_test(self, state, goal):
        return state == goal

    def get_cost(self, state1, state2):
        return 1
    
    # FIX: Added 'self' so the agent can call this function
    def get_value(self, node):
        """Objective function: Returns the numeric grade or 0 for non-grade nodes."""
        try:
            if 'G3' in node:
                return int(node.split('_')[1])
            return 0
        except:
            return 0

    # --- Step 2 & 3: Searches ---
    def bfs(self, start, goal):
        start_time = time.time()
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            current_node, path = queue.popleft()
            if self.goal_test(current_node, goal):
                return path, (time.time() - start_time) * 1000
            for neighbor in self.perceive(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, 0

    def dfs(self, start, goal):
        start_time = time.time()
        stack = [(start, [start])]
        visited = set()
        while stack:
            current_node, path = stack.pop()
            if self.goal_test(current_node, goal):
                return path, (time.time() - start_time) * 1000
            if current_node not in visited:
                visited.add(current_node)
                for neighbor in self.perceive(current_node):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        return None, 0

    def heuristic(self, current, goal):
        return abs(self.get_value(goal) - self.get_value(current))

    def ucs(self, start, goal):
        start_time = time.time()
        pq = [(0, start, [start])]
        visited = {}
        while pq:
            cost, current_node, path = heapq.heappop(pq)
            if self.goal_test(current_node, goal):
                return path, (time.time() - start_time) * 1000, cost
            if current_node not in visited or cost < visited[current_node]:
                visited[current_node] = cost
                for neighbor in self.perceive(current_node):
                    new_cost = cost + self.get_cost(current_node, neighbor)
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
        return None, 0, 0

    def a_star(self, start, goal):
        start_time = time.time()
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        visited = {}
        while pq:
            f_score, g_score, current_node, path = heapq.heappop(pq)
            if self.goal_test(current_node, goal):
                return path, (time.time() - start_time) * 1000, g_score
            if current_node not in visited or g_score < visited[current_node]:
                visited[current_node] = g_score
                for neighbor in self.perceive(current_node):
                    new_g = g_score + self.get_cost(current_node, neighbor)
                    new_f = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        return None, 0, 0

    # --- Step 4: Local Searches ---
    def hill_climbing(self, start):
        current = start
        while True:
            neighbors = self.perceive(current)
            if not neighbors: break
            best_neighbor = max(neighbors, key=self.get_value)
            if self.get_value(best_neighbor) > self.get_value(current):
                current = best_neighbor
            else: break
        return current

    def simulated_annealing(self, start):
        current = start
        temp = 100.0
        while temp > 0.01:
            neighbors = self.perceive(current)
            if not neighbors: break
            next_node = random.choice(neighbors)
            delta = self.get_value(next_node) - self.get_value(current)
            if delta > 0 or random.uniform(0, 1) < math.exp(delta / temp):
                current = next_node
            temp *= 0.95
        return current

    def local_beam_search(self, k):
        """FIX: Added missing Beam Search method"""
        nodes = random.sample(list(self.graph.keys()), k)
        for _ in range(5):
            all_neighbors = []
            for n in nodes:
                all_neighbors.extend(self.perceive(n))
            if not all_neighbors: break
            nodes = sorted(list(set(all_neighbors)), key=self.get_value, reverse=True)[:k]
        return nodes[0]

    # --- Step 5: Adversarial Search ---
    def minimax(self, node, depth, is_maximizing):
        self.nodes_evaluated += 1
        if depth == 0 or not self.perceive(node):
            return self.get_value(node)
        neighbors = self.perceive(node)
        if is_maximizing:
            return max(self.minimax(n, depth - 1, False) for n in neighbors)
        else:
            return min(self.minimax(n, depth - 1, True) for n in neighbors)

    def alpha_beta(self, node, depth, alpha, beta, is_maximizing):
        self.nodes_evaluated += 1
        if depth == 0 or not self.perceive(node):
            return self.get_value(node)
        neighbors = self.perceive(node)
        if is_maximizing:
            v = float('-inf')
            for n in neighbors:
                v = max(v, self.alpha_beta(n, depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if alpha >= beta: break
            return v
        else:
            v = float('inf')
            for n in neighbors:
                v = min(v, self.alpha_beta(n, depth - 1, alpha, beta, True))
                beta = min(beta, v)
                if alpha >= beta: break
            return v
      
    
    def is_consistent(self, var, value, assignment):
     """
     Checks if a value for a variable is consistent with current assignments
      based on the constraints defined in Phase 3.
     """
     for (v, val) in assignment.items():
        if not self.check_constraint(var, value, v, val):
            return False
     return True
# --- Demonstration Block ---
if __name__ == "__main__":
    agent = AIAgent(graph)
    start_node = 'ST_1'
    goal_node = 'G3_18'
    
    # Calculate global max for Hill Climbing test
    all_values = [agent.get_value(n) for n in graph.keys()]
    global_max_val = max(all_values)
    global_max_node = f"G3_{global_max_val}"

    print(f"--- PHASES 2-5: SEARCH EVALUATION ---\n")

    # [STEP 2 & 3] Pathfinding
    print(f"1. Testing Pathfinding (Target: {goal_node})")
    for name, method in [("BFS", agent.bfs), ("DFS", agent.dfs), ("UCS", agent.ucs), ("A*", agent.a_star)]:
        result = method(start_node, goal_node)
        path, t = result[0], result[1]
        print(f"   {name:<5} | Time: {t:.3f}ms | Path Len: {len(path) if path else 'N/A'}")

    # [STEP 4] Local Search
    print(f"\n2. Testing Local Search")
    hc_results = [agent.hill_climbing(random.choice(list(graph.keys()))) for _ in range(10)]
    successes = sum(1 for r in hc_results if agent.get_value(r) == global_max_val)
    print(f"   Hill Climbing (10 runs): {successes} global peaks reached ({successes*10}%)")
    
    sa_res = agent.simulated_annealing(start_node)
    print(f"   Simulated Annealing Result: {sa_res} (Value: {agent.get_value(sa_res)})")
    
    print(f"   Beam Search: k=3 -> {agent.local_beam_search(3)} | k=5 -> {agent.local_beam_search(5)}")

    # [STEP 5] Adversarial Search
    print(f"\n3. Testing Adversarial Search (Depth 4)")
    agent.nodes_evaluated = 0
    m_val = agent.minimax(start_node, 4, True)
    m_nodes = agent.nodes_evaluated
    
    agent.nodes_evaluated = 0
    ab_val = agent.alpha_beta(start_node, 4, float('-inf'), float('inf'), True)
    ab_nodes = agent.nodes_evaluated
    
    print(f"   Minimax: Value {m_val} | Nodes: {m_nodes}")
    print(f"   Alpha-Beta: Value {ab_val} | Nodes: {ab_nodes}")
    print(f"   Efficiency: Pruning saved {m_nodes - ab_nodes} nodes.")