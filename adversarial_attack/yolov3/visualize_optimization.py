from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
def main():


    fig = plt.figure()
    ax = Axes3D(fig)
    df = pd.read_csv('/Users/ian/adversarial_attack/optimization_results/combined_optimization_results')
    lr = df.lr.to_list()
    generations = df.num_generations.to_list()
    confidence = df.confidence.to_list()


    ax.scatter(lr, generations, confidence)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Axes3D.scatter(lr, generations, confidence, zdir='z', s=20, c=None, depthshade=True)


main()