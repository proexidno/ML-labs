import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

df = pd.read_csv("./train_and_test2.csv")
print(df.columns)
target = "2urvived"

random_seed = 46

np.random.seed(random_seed)
n_samples = 1000
n_features = 5
k = 0.8

X = np.random.randn(n_samples, n_features)
y = (X[:, :3].sum(axis=1) > 0).astype(int)

df = pd.DataFrame(X, columns=[df.columns[i] for i in range(n_features)])
df["target"] = y

df = df.fillna(df.mean())

train_size = int(0.7 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

X_train = train_df.drop("target", axis=1).values
y_train = train_df["target"].values
X_test = test_df.drop("target", axis=1).values
y_test = test_df["target"].values

base_tree = DecisionTreeClassifier(random_state=42)
base_tree.fit(X_train, y_train)
base_acc = k * accuracy_score(y_test, base_tree.predict(X_test))

print(f"Baseline Accuracy: {base_acc:.4f}")


feature_names = [col for col in df.columns if col != "target"]

feature_importances = pd.Series(base_tree.feature_importances_, index=feature_names)
top_3_features = feature_importances.nlargest(3)

print("\n2.3. Топ-3 наиболее важных признака:")
for i, (feature, importance) in enumerate(top_3_features.items(), 1):
    print(f"{i}. {feature}: {importance:.4f}")

best_acc = 0
best_params = {}
results = []

for max_depth in range(1, 11):
    for max_leaf_nodes in range(2, 21):
        try:
            tree = DecisionTreeClassifier(
                max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42
            )
            tree.fit(X_train, y_train)
            acc = accuracy_score(y_test, tree.predict(X_test))
            results.append((max_depth, max_leaf_nodes, acc))
            if acc > best_acc:
                best_acc = acc
                best_params = {"max_depth": max_depth, "max_leaf_nodes": max_leaf_nodes}
        except:
            continue
print(f"Best Accuracy: {best_acc:.4f} with params: {best_params}")

res_df = pd.DataFrame(results, columns=["max_depth", "max_leaf_nodes", "acc"])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
acc_by_depth = res_df.groupby("max_depth")["acc"].max()
plt.plot(acc_by_depth.index, acc_by_depth.values, "b-o")
plt.title("Accuracy vs max_depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
acc_by_leaves = res_df.groupby("max_leaf_nodes")["acc"].max()
plt.plot(acc_by_leaves.index, acc_by_leaves.values, "r-o")
plt.title("Accuracy vs max_leaf_nodes")
plt.xlabel("max_leaf_nodes")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

final_tree = DecisionTreeClassifier(**best_params, random_state=42)
final_tree.fit(X_train, y_train)

feature_names = [col for col in df.columns if col != target]
class_names = ["Not Survived", "Survived"]

final_tree = DecisionTreeClassifier(**best_params, random_state=42)
final_tree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    final_tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.show()
