import pandas as pd


#Step 3: Load and Inspect the Dataset
df= pd.read_csv('student-mat.csv', sep=';')# our datatset is separated by semicolon instead of colon
print(df.head(10))
print(f"Shape of Dataset: {df.shape}")
print(f" Types of dataset: {df.dtypes}")
print(f" Number of missing values in dataset: {df.isnull().sum()}")
print(f"Number of unique values in target column: {df['G3'].nunique()}")

#Step 4: Apply Python Fundamentals

# 7. Use a for loop to count occurrences in the target column (G3)
# and store the result in a Python dictionary [cite: 68, 69]
class_distribution = {}
for grade in df['G3']:
    if grade in class_distribution:
        class_distribution[grade] += 1
    else:
        class_distribution[grade] = 1

# 8. Use a list to store the names of all columns in the dataset 
column_names = list(df.columns)

# 9. Use slicing to extract the first 50 rows 
sample_data = df[:50]

# 10. Write a function called describe_dataset() 
def describe_dataset():
    """
    Prints the shape, column names, and class distribution of the dataset.
    """
    print("--- Dataset Description ---")
    print(f"Shape: {df.shape}") # 
    print(f"Column Names: {column_names}") # 
    print(f"Class Distribution (G3 Grade: Count):")
    # Sorting the dictionary by grade (key) for better readability
    for grade in sorted(class_distribution.keys()):
        print(f"  Grade {grade}: {class_distribution[grade]} students") # 

# Call the function to verify the output
describe_dataset()

#step5:
class DataRecord:
    def __init__(self, record_id, features, label):
        self.record_id = record_id  # Unique identifier for the row [cite: 76]
        self.features = features    # A dictionary of input variables [cite: 77]
        self.label = label
    
    def display(self):
        print(f"Record ID: {self.record_id}")
        print(f"Features: {self.features}")
        print(f"Label (Target): {self.label}")
        print("-"*20)

records=[]

for i in range(5):
    row= df.iloc[i]

    feat_dict= row.drop('G3').to_dict()
    target_label= row['G3']

    new_record= DataRecord(record_id=i, features=feat_dict, label=target_label)
    records.append(new_record)

for r in records:
    r.display()


#Step 6: Build a Graph from Your Data

# 11 & 12. Identify columns and extract unique nodes
col1 = 'studytime'
col2 = 'G3'

# Create a set of all unique values to represent nodes
nodes_st = df[col1].unique()
nodes_g3 = df[col2].unique()

# We use strings to distinguish nodes if values overlap (e.g., grade 2 vs studytime 2)
all_nodes = [f"ST_{v}" for v in nodes_st] + [f"G3_{v}" for v in nodes_g3]

# 13. Create the graph dictionary (Adjacency List)
graph = {node: set() for node in all_nodes}

for _, row in df.iterrows():
    u = f"ST_{row[col1]}"
    v = f"G3_{row[col2]}"
    
    # Add an undirected edge between the study habit and the grade
    graph[u].add(v)
    graph[v].add(u)

# Convert sets back to lists for standard dictionary printing
graph_list = {k: list(v) for k, v in graph.items()}

# 14. Print the graph dictionary
print("--- Graph Adjacency List ---")
import pprint
pprint.pprint(graph_list)

# 15. Count and print total nodes and edges
total_nodes = len(graph)
# Sum all connections and divide by 2 for undirected edges
total_edges = sum(len(v) for v in graph.values()) // 2

print(f"\nTotal Nodes: {total_nodes}")
print(f"Total Edges: {total_edges}") 