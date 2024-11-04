from imports import *
from pre_process import preprocess_data
from hyperparamter_tuning import plot_hyperparameter_results
from metrics import plot_confusion_matrix,calculate_metrics

class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}
        
    def calculate_gini(self, child_nodes):
        n = 0
        for node in child_nodes:

            n = n + len(node)
        gini = 0
        for node in child_nodes:
            m = len(node)
            if m == 0:
                continue
            y = []
            for row in node:
                y.append(row[-1])
            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2
            gini = gini + (m / n) * node_gini
        return gini


    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child

    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_score = 1000
        f_index = 0
        f_value = 0
        for column in range(num_of_features):
            for row in data:
                value = row[column]
                l, r = self.apply_split(column, value, data)
                children = [l, r]
                score = self.calculate_gini(children)
                if score < gini_score:
                    gini_score = score
                    f_index = column
                    f_value = value
                    child_nodes = children
        node = {"feature": f_index, "value": f_value, "children": child_nodes}
        return node


    def calc_class(self, node):
        y = []
        for row in node:
            y.append(row[-1])
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]


    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        if depth >= self.max_depth:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    def train(self, X):
        tree = self.find_best_split(X)
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    def print_dt(self, tree, depth=0,branch="root"):
        indent = "    " * depth  
        if "feature" in tree:
            if branch == "root":
                print(f"{indent}Root Node: [feature #{tree['feature']} < {tree['value']}]")
            elif branch == "left":
                print(f"{indent}├── Left: [feature #{tree['feature']} < {tree['value']}]")
            else:
                print(f"{indent}└── Right: [feature #{tree['feature']} < {tree['value']}]")
        
            self.print_dt(tree["left"], depth + 1, branch="left")
            self.print_dt(tree["right"], depth + 1, branch="right")
        else:
            if branch == "left":
                print(f"{indent}├── Left (Leaf): class = {tree['class_value']}, depth = {tree['depth']}")
            else:
                print(f"{indent}└── Right (Leaf): class = {tree['class_value']}, depth = {tree['depth']}")

    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)

def hyperparameter_tuning_fn(X_train, y_train, X_test, y_test, max_depth_values):
    results = np.zeros(len(max_depth_values))

    for i, max_depth in enumerate(max_depth_values):
        dt = DecisionTree(max_depth,3)
        dt.train(X_train)
        y_pred = dt.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        results[i] = accuracy

    return results

if __name__ == "__main__":
    print("running......")
    data = pd.read_csv('./Data/diabetes.csv')
    x_train, x_test, y_train, y_test = preprocess_data(data, target_column="Outcome")
    train_data = x_train.to_numpy();train_y = y_train.to_numpy();test_data = x_test.to_numpy();test_y = y_test.to_numpy()
    
    max_depth_values = range(4,10)  

    results = hyperparameter_tuning_fn(train_data, train_y, test_data, test_y, max_depth_values)
    plot_hyperparameter_results(results, max_depth_values)

    dt = DecisionTree(6, 2)
    tree = dt.train(train_data)
    y_pred = dt.predict(train_data)
    y_pred_test = dt.predict(test_data)
    
    
    print(f"Accuracy for the train data: {sum(y_pred == train_y) / train_y.shape[0]}")
    print(f"Accuracy for the test data: {sum(y_pred_test == test_y) / test_y.shape[0]}")
    precision_train, recall_train, f1_train = calculate_metrics(train_y, y_pred)
    precision_test, recall_test, f1_test = calculate_metrics(test_y, y_pred_test)

    print(f"Train Precision: {precision_train}, Train Recall: {recall_train}, Train F1 Score: {f1_train}")
    print(f"Test Precision: {precision_test}, Test Recall: {recall_test}, Test F1 Score: {f1_test}")

    plot_confusion_matrix(train_y, y_pred)

    dt.print_dt(tree)