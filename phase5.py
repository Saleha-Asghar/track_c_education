import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from phase1 import df  # Use your existing dataframe

# --- Step 47: Handle Missing Values ---
# Even though Phase 1 showed 0 missing, the manual requires this logic for robustness
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].mean()) # Replace numerical with mean
    else:
        df[col] = df[col].fillna(df[col].mode()[0]) # Replace categorical with most frequent

# --- Step 48: Encode Categorical Variables ---
# Convert text categories (like 'school', 'sex', 'address') into numbers
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# --- Step 49: Normalize Numerical Features ---
# Target variable G3 should remain as is, but features should be normalized
target_col = 'G3'
features = df_encoded.drop(columns=[target_col])
target = df_encoded[target_col]

# Subtract mean and divide by standard deviation
features_normalized = (features - features.mean()) / features.std()

# --- Step 50: Split Dataset (80/20) ---
# Using a fixed seed (42) for reproducibility
np.random.seed(42)
indices = np.random.permutation(len(df_encoded))
train_size = int(0.8 * len(df_encoded))

train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = features_normalized.iloc[train_idx], features_normalized.iloc[test_idx]
y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]


# --- Step 2: Unsupervised Learning (K-Means from Scratch) ---

def get_euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

def k_means(data, k, max_iterations=100):
    # Initialize k centroids by randomly selecting k rows
    centroids = data.sample(n=k).values
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        
        # Assign each row to the nearest centroid
        for idx, row in data.iterrows():
            distances = [get_euclidean_distance(row.values, c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)
            
        # Update each centroid by computing the mean of the cluster
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(data.loc[clusters[i]].mean().values)
            else:
                new_centroids.append(centroids[i]) # Keep old if cluster empty
                
        new_centroids = np.array(new_centroids)
        
        # Check if centroids have stopped changing
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return clusters, centroids
def get_total_distance(medoid_idx, cluster_indices, data):
    # Calculates sum of distances from a candidate medoid to all other points in the cluster
    medoid_point = data.loc[medoid_idx].values
    total_dist = 0
    for idx in cluster_indices:
        total_dist += get_euclidean_distance(medoid_point, data.loc[idx].values)
    return total_dist

def k_medoids(data, k, max_iterations=50):
    # [Step 59] Initialize by randomly selecting k rows as medoids
    medoid_indices = random.sample(list(data.index), k)
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        
        # [Step 60] Assign each row to the nearest medoid
        for idx, row in data.iterrows():
            distances = [get_euclidean_distance(row.values, data.loc[m_idx].values) 
                         for m_idx in medoid_indices]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)
            
        # [Step 61] Update medoids: find the point that minimizes total distance in cluster
        new_medoid_indices = []
        for i in range(k):
            if clusters[i]:
                # Find point in cluster with minimum sum of distances to others
                best_medoid = min(clusters[i], 
                                  key=lambda idx: get_total_distance(idx, clusters[i], data))
                new_medoid_indices.append(best_medoid)
            else:
                new_medoid_indices.append(medoid_indices[i])
        
        # [Step 62] Repeat until medoids do not change[cite: 1]
        if set(new_medoid_indices) == set(medoid_indices):
            break
        medoid_indices = new_medoid_indices
        
    return clusters, medoid_indices

def calculate_wcsd(clusters, centers, data, is_kmeans=True):
    # [Step 63] Within-Cluster Sum of Distances[cite: 1]
    total_wcsd = 0
    for i, cluster in enumerate(clusters):
        center = centers[i] if is_kmeans else data.loc[centers[i]].values
        for idx in cluster:
            total_wcsd += get_euclidean_distance(data.loc[idx].values, center)
    return total_wcsd

# --- Step 3: Perceptron Training from Scratch ---

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.history = []

    def fit(self, X, y):
        # [Step 64] Initialize weights as zeros (one per feature)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            errors = 0
            for idx, x_i in enumerate(X.values):
                # [Step 65] Compute weighted sum
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # [Step 66] Step Activation Function
                y_predicted = 1 if linear_output > 0 else 0
                
                # [Step 67] Compute error
                error = y[idx] - y_predicted
                
                # [Step 68] Update weights and bias
                if error != 0:
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += 1
            
            # [Step 69] Record accuracy for this epoch
            accuracy = 1 - (errors / n_samples)
            self.history.append(accuracy)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output > 0, 1, 0)
    

# --- Step 4: Delta Rule and Gradient Descent from Scratch ---

class DeltaRuleModel:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.mse_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # [Step 71] Initialize weights as random small values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for epoch in range(self.epochs):
            weight_updates = np.zeros(n_features)
            bias_update = 0
            epoch_error = 0
            
            for idx, x_i in enumerate(X.values):
                # [Step 72] Compute linear output (dot product)
                output = np.dot(x_i, self.weights) + self.bias
                
                # [Step 73] Compute error (True - Predicted)
                error = y[idx] - output
                epoch_error += error**2
                
                # [Step 74 & 75] Accumulate updates for Batch Gradient Descent
                weight_updates += error * x_i
                bias_update += error
            
            # [Step 75] Apply averaged weight updates
            self.weights += (self.lr * weight_updates) / n_samples
            self.bias += (self.lr * bias_update) / n_samples
            
            # [Step 76] Record Mean Squared Error
            self.mse_history.append(epoch_error / n_samples)


# --- Step 5: MLP with Backpropagation from Scratch ---

class MultilayerPerceptron:
    def __init__(self, input_size, hidden1=16, hidden2=8, output_size=1, lr=0.01, epochs=200):
        # Weight Initialization (He Initialization for ReLU)
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2./hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        self.acc_history = []

    # --- Activation Functions & Derivatives ---
    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            # [Step 78-83] Forward Propagation
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.relu(z1)
            
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.relu(z2)
            
            z3 = np.dot(a2, self.W3) + self.b3
            a3 = self.sigmoid(z3) # Output probability

            # [Step 84-87] Backpropagation
            # Output Layer Error
            dz3 = a3 - y
            dW3 = np.dot(a2.T, dz3) / n_samples
            db3 = np.sum(dz3, axis=0, keepdims=True) / n_samples

            # Hidden Layer 2 Error
            dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(z2)
            dW2 = np.dot(a1.T, dz2) / n_samples
            db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

            # Hidden Layer 1 Error
            dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(z1)
            dW1 = np.dot(X.T, dz1) / n_samples
            db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

            # Update Weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3

            # [Step 88] Record Loss and Accuracy every 10 epochs
            if epoch % 10 == 0:
                loss = self.binary_cross_entropy(y, a3)
                preds = (a3 > 0.5).astype(int)
                acc = np.mean(preds == y)
                self.loss_history.append(loss)
                self.acc_history.append(acc)

    def predict(self, X):
        a1 = self.relu(np.dot(X, self.W1) + self.b1)
        a2 = self.relu(np.dot(a1, self.W2) + self.b2)
        return (self.sigmoid(np.dot(a2, self.W3) + self.b3) > 0.5).astype(int)





# --- Step 6: Final Evaluation and Comparison ---

def evaluate_models():
    print("\n" + "="*40)
    print("STEP 6: FINAL MODEL COMPARISON")
    print("="*40)
    
    # 1. Perceptron Performance
    p_preds = model.predict(X_test)
    p_acc = np.mean(p_preds == y_test_bin)
    
    # 2. Delta Rule Performance
    # For Delta Rule (linear), we use 0.5 as the threshold for Pass/Fail
    dr_preds = (np.dot(X_test, delta_model.weights) + delta_model.bias > 0.5).astype(int)
    dr_acc = np.mean(dr_preds == y_test_bin)
    
    # 3. MLP Performance
    mlp_acc = np.mean(mlp_test_preds == y_test_bin.reshape(-1,1))
    
    # --- Comparison Table ---
    data = {
        "Model": ["Perceptron", "Delta Rule", "MLP"],
        "Architecture": ["Single Neuron (Step)",   "    Single Neuron (Linear)", "    3-Layer (ReLU/Sigmoid)"],
            "        Test Accuracy": [f"{p_acc:.2%}", f"{dr_acc:.2%}", f"{mlp_acc:.2%}"],
        "Best Use Case": ["Linearly Separable Data", "       Linear Regression/Simple Class", "Complex Non-Linear Patterns"]
    }
    
    comparison_df = pd.DataFrame(data)
    print(comparison_df.to_string(index=False))




if __name__ == "__main__":
    print("--- PHASE 4: MACHINE LEARNING PREPARATION ---")
    print(f" Training set shape: {X_train.shape}")
    print(f" Testing set shape: {X_test.shape}")
    print("Data preparation complete and synchronized!")
    k_value = df['G3'].nunique() 
    print(f"\nStep 2: Running K-Means with k={k_value}...")
    
    clusters, final_centroids = k_means(X_train, k_value)
    
    print(f"K-Means complete. Found {len(clusters)} clusters.")
    # Measure cluster purity (manual requirement)
    sample_cluster = clusters[0]
    if sample_cluster:
        most_common_label = y_train.loc[sample_cluster].mode()[0]
        purity = (y_train.loc[sample_cluster] == most_common_label).mean()
        print(f"Sample Cluster Purity: {purity:.2%}")

       
    k_val = 5 # Using a smaller k for faster local testing
    
    # Run K-Means
    km_clusters, km_centroids = k_means(X_train, k_val)
    km_wcsd = calculate_wcsd(km_clusters, km_centroids, X_train, is_kmeans=True)
    
    # Run K-Medoids
    kmed_clusters, kmed_indices = k_medoids(X_train, k_val)
    kmed_wcsd = calculate_wcsd(kmed_clusters, kmed_indices, X_train, is_kmeans=False)
    
    print(f"\nStep 63: Clustering Quality Comparison (k={k_val})")
    print(f"K-Means WCSD: {km_wcsd:.2f}")
    print(f"K-Medoids WCSD: {kmed_wcsd:.2f}")

    ##step #3: Perceptron Training
    # Prepare binary labels for Education Track (Pass/Fail)
    y_train_bin = (y_train >= 10).astype(int).values
    y_test_bin = (y_test >= 10).astype(int).values

    print("\nStep 3: Training Perceptron...")
    model = Perceptron(learning_rate=0.01, epochs=50)
    model.fit(X_train, y_train_bin)

    # [Step 70] Plotting Training Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 51), model.history, marker='o', color='b')
    plt.title('Step 70: Perceptron Training Accuracy over 50 Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == y_test_bin)
    print(f"Perceptron Test Accuracy: {test_acc:.2%}")

    print("\nStep 4: Training Delta Rule (Batch Gradient Descent)...")
    delta_model = DeltaRuleModel(learning_rate=0.05, epochs=100)
    delta_model.fit(X_train, y_train_bin)

    # [Step 77] Comparison Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), delta_model.mse_history, color='red', label='Delta Rule MSE')
    plt.title('Step 76 & 77: Delta Rule Convergence (MSE Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Final MSE after 100 epochs: {delta_model.mse_history[-1]:.4f}")


    print("\nStep 5: Training Multilayer Perceptron (MLP)...")
    mlp = MultilayerPerceptron(input_size=X_train.shape[1], lr=0.1, epochs=200)
    mlp.fit(X_train.values, y_train_bin)

    # Plotting Loss and Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, 200, 10), mlp.loss_history, color='orange', label='BCE Loss')
    plt.title('MLP Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, 200, 10), mlp.acc_history, color='green', label='Accuracy')
    plt.title('MLP Training Accuracy')
    plt.legend()
    plt.show()

    mlp_test_preds = mlp.predict(X_test.values)
    print(f"MLP Test Accuracy: {np.mean(mlp_test_preds == y_test_bin.reshape(-1,1)):.2%}")

    evaluate_models()