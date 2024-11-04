from imports import *
def plot_hyperparameter_results(results, max_depth_values):
    plt.figure(figsize=(12, 6))
    plt.plot(max_depth_values, results, marker='o', color='blue')
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter Tuning Results: Accuracy')
    plt.xticks(max_depth_values)

    plt.ylim(0.65, 0.8) 
    plt.grid()
    plt.savefig("./graphs/hyperparamter-tuning.png", format='png', dpi=300)
    plt.savefig("./assets/hyperparamter-tuning.png", format='png', dpi=300)
    plt.show()
