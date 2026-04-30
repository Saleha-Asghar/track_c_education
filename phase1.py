import pandas as pd
import pprint

# --- Step 3: Load and Inspect the Dataset ---
df = pd.read_csv('student-mat.csv', sep=';') 

# --- Step 4: Apply Python Fundamentals ---
# Using a for loop to count occurrences in the target column (G3)
class_distribution = {}
for grade in df['G3']:
    if grade in class_distribution:
        class_distribution[grade] += 1
    else:
        class_distribution[grade] = 1

column_names = list(df.columns)
sample_data = df[:50]

def describe_dataset():
    """Prints the shape, column names, and class distribution of the dataset."""
    print("--- Dataset Description ---")
    print(f"Shape: {df.shape}") 
    print(f"Column Names: {column_names}") 
    print(f"Class Distribution (G3 Grade: Count):")
    for grade in sorted(class_distribution.keys()):
        print(f"  Grade {grade}: {class_distribution[grade]} students") 

# --- Step 5: DataRecord Class ---
class DataRecord:
    def __init__(self, record_id, features, label):
        self.record_id = record_id  # Unique identifier
        self.features = features    # Dictionary of input variables
        self.label = label
    
    def display(self):
        print(f"Record ID: {self.record_id}")
        print(f"Features: {self.features}")
        print(f"Label (Target): {self.label}")
        print("-" * 20)

# --- Step 6: Build a Graph from Your Data ---
col1 = 'studytime'
col2 = 'G3'

nodes_st = df[col1].unique()
nodes_g3 = df[col2].unique()
all_nodes = [f"ST_{v}" for v in nodes_st] + [f"G3_{v}" for v in nodes_g3]

# Create the graph dictionary (Adjacency List)
graph = {node: set() for node in all_nodes}

for _, row in df.iterrows():
    u = f"ST_{row[col1]}"
    v = f"G3_{row[col2]}"
    graph[u].add(v)
    graph[v].add(u)

# Convert sets to lists for standard dictionary use
graph_list = {k: list(v) for k, v in graph.items()}

# Calculate totals for metadata
total_nodes = len(graph)
total_edges = sum(len(v) for v in graph.values()) // 2

# --- EXECUTION BLOCK ---
# Everything inside here ONLY runs when you play this file directly.
if __name__ == "__main__":
    print("--- Phase 1 Data Preview ---")
    print(df.head(10))
    print(f"Shape of Dataset: {df.shape}")
    print(f"Number of missing values: {df.isnull().sum().sum()}")
    print(f"Number of unique values in target column: {df['G3'].nunique()}")

    print("\nCalling describe_dataset():")
    describe_dataset()

    print("\n--- Sample DataRecords ---")
    records = []
    for i in range(5):
        row = df.iloc[i]
        feat_dict = row.drop('G3').to_dict()
        new_record = DataRecord(record_id=i, features=feat_dict, label=row['G3'])
        records.append(new_record)

    for r in records:
        r.display()

    print("\n--- Graph Adjacency List ---")
    pprint.pprint(graph_list)

    print(f"\nTotal Nodes: {total_nodes}")
    print(f"Total Edges: {total_edges}") 
    print("Graph built successfully!")