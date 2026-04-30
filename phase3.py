import pandas as pd
import random
from phase1 import df  # Import your loaded dataframe
from phase2 import AIAgent # Import your agent structure

# --- Step 1: Define CSP Components ---
## Formal CSP DefinitionVariables (X): The unknowns are the student profile attributes required to reach a specific academic outcome. 
#  G3: The final grade.  
# absences: The number of school days missed.  
# studytime: Weekly study hours.  
# Domains (D): The possible values extracted from the student-mat.csv. 
#  G3: \{0, 1, 2, \dots, 20\}. 
#  absences: \{0, 1, 2, \dots, 93\}. 
#  studytime: \{1, 2, 3, 4\}. 
#  Constraints (C):Here are the five rules we have implemented in the logic: 
# 1 If absences are greater than 20, the G3 grade cannot be in the top-tier range (above 15). 
# 2 If studytime is at the minimum level (1), a perfect or near-perfect G3 grade (19-20) is inconsistent. 
# 3 If studytime is at the maximum level (4), a failing G3 grade (0-5) is inconsistent unless other negative factors are present. 
# 4 A G3 grade of 20 requires a studytime of at least level 3 or 4. 
# 5 High absences (above 30) are inconsistent with a studytime of level 4 (as high attendance is usually a prerequisite for intensive study habits). 
# Goal: A valid complete assignment is a set of values \{G3: g, absences: a, studytime: s\} where all five logical rules are satisfied simultaneously.
VARIABLES = ['G3', 'absences', 'studytime']
DOMAINS = {
    'G3': sorted(df['G3'].unique().tolist()),
    'absences': sorted(df['absences'].unique().tolist()),
    'studytime': sorted(df['studytime'].unique().tolist())
}

# --- Step 2: Define Constraints ---
def check_constraint(var1, val1, var2, val2):
    """
    Defines the rules for a valid student profile.
    Returns True if the combination of values is allowed.
    """
    # Rule 1: High absences (>20) usually don't result in top grades (>15)
    if var1 == 'absences' and var2 == 'G3':
        if val1 > 20 and val2 > 15: return False
    
    # Rule 2: Low study time (1) is inconsistent with very high grades (19-20)
    if var1 == 'studytime' and var2 == 'G3':
        if val1 == 1 and val2 >= 19: return False
        
    # Rule 3: High study time (4) is inconsistent with failing grades (0-5) 
    # unless absences are also very high (this adds complexity)
    if var1 == 'studytime' and var2 == 'G3':
        if val1 == 4 and val2 <= 5: return False

    return True

def is_consistent(var, value, assignment):
     
    for (v, val) in assignment.items():
        if not check_constraint(var, value, v, val):
            return False
    return True

# --- Step 3: AC-3 Algorithm (Constraint Propagation) ---
def ac3(domains):
    queue = [(v1, v2) for v1 in VARIABLES for v2 in VARIABLES if v1 != v2]
    while queue:
        (xi, xj) = queue.pop(0)
        if revise(domains, xi, xj):
            if not domains[xi]: return False
            for xk in [v for v in VARIABLES if v != xi and v != xj]:
                queue.append((xk, xi))
    return True

def revise(domains, xi, xj):
    revised = False
    for x in domains[xi][:]:
        # If no value in xj's domain satisfies the constraint with x
        if not any(check_constraint(xi, x, xj, y) for y in domains[xj]):
            domains[xi].remove(x)
            revised = True
    return revised

# --- Step 4: Backtracking with MRV Heuristic ---
# def select_unassigned_variable(assignment, domains):
#     # MRV: Choose variable with the fewest remaining values
#     unassigned = [v for v in VARIABLES if v not in assignment]
#     return min(unassigned, key=lambda v: len(domains[v]))

backtrack_count = 0

def backtracking_search(assignment, domains, mode="basic"):
    global backtrack_count
    if len(assignment) == len(VARIABLES):
        return assignment
    
    # --- MRV Heuristic Selection ---
    if mode == "mrv":
        unassigned = [v for v in VARIABLES if v not in assignment]
        var = min(unassigned, key=lambda v: len(domains[v]))
    else:
        # Basic: Just pick the next unassigned variable in fixed order
        var = [v for v in VARIABLES if v not in assignment][0]

    for value in domains[var]:
        if is_consistent(var, value, assignment):
            assignment[var] = value
            
            # --- Forward Checking Logic ---
            if mode == "forward_checking" or mode == "mrv":
                local_domains = {v: list(d) for v, d in domains.items()}
                # Prune domains of other variables based on this assignment
                for other_var in [v for v in VARIABLES if v not in assignment]:
                    local_domains[other_var] = [val for val in local_domains[other_var] 
                                               if check_constraint(var, value, other_var, val)]
                    if not local_domains[other_var]: # Empty domain found!
                        backtrack_count += 1
                        del assignment[var]
                        continue 

            result = backtracking_search(assignment, domains, mode)
            if result: return result
            
            # Backtracking happens here
            backtrack_count += 1
            del assignment[var]
            
    return None


def min_conflicts(variables, domains, constraints_func, max_steps=100):
    # [Step 38] Start with a random complete assignment
    current = {v: random.choice(domains[v]) for v in variables}
    violation_history = []
    
    for i in range(max_steps):
        # [Step 39] Count the number of violated constraints
        conflicts = []
        for v in variables:
            # Check if this variable violates any constraints with others
            has_conflict = any(not constraints_func(v, current[v], other, current[other]) 
                               for other in variables if v != other)
            if has_conflict:
                conflicts.append(v)
        
        # Record number of violations for Step 43
        violation_history.append(len(conflicts))
        
        # [Step 42] Repeat until no constraints are violated
        if not conflicts:
            return current, i, violation_history
        
        # [Step 40] Select a variable involved in at least one violation
        var = random.choice(conflicts)
        
        # [Step 41] Assign value that minimizes violated constraints
        def count_var_conflicts(val):
            return sum(1 for other in variables if var != other 
                       and not constraints_func(var, val, other, current[other]))
        
        best_val = min(domains[var], key=count_var_conflicts)
        current[var] = best_val
        
    return current, max_steps, violation_history
    
# --- Execution ---
if __name__ == "__main__":
    print("--- PHASE 3: CONSTRAINT SATISFACTION ---")
    
    # Test AC-3
    test_domains = {v: list(d) for v, d in DOMAINS.items()}
    print(f"Original Domain Sizes: {[(v, len(test_domains[v])) for v in VARIABLES]}")
    
    if ac3(test_domains):
        print(f"Post-AC3 Domain Sizes: {[(v, len(test_domains[v])) for v in VARIABLES]}")
    
    # Test Backtracking
   # [Step 3] Backtracking Comparisons
    modes = ["basic", "forward_checking", "mrv"]
    for mode in modes:
        
        backtrack_count = 0
        current_domains = {v: list(d) for v, d in test_domains.items()}
        sol = backtracking_search({}, current_domains, mode)
        print(f"Step 3: {mode.upper()} Solution: {sol} | Backtracks: {backtrack_count}")
        # [Step 4] Local Search (Min-Conflicts)
    sol_mc, steps, history = min_conflicts(VARIABLES, DOMAINS, check_constraint)
    print(f"\nStep 4: Min-Conflicts Result after {steps} iterations: {sol_mc}")
    print(f"Step 4: Violation History (first 10 steps): {history[:10]}")