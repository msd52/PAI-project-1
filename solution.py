import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import KMeans

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0




from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, DotProduct

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        first_kernel = ConstantKernel()*Matern()+ConstantKernel()*RBF()+ConstantKernel()*DotProduct()#ConstantKernel(constant_value=0.0, constant_value_bounds=(1e-05, 100000.0))
        self.gaussian_process = GaussianProcessRegressor(kernel=first_kernel, n_restarts_optimizer=5, normalize_y=True, random_state=0)
        # self.gp_2dlist = []
        # for x in range(0,2):
        #     self.gp_2dlist.append([])
        #     for y in range(0,2):
        #         self.gp_2dlist[-1].append(GaussianProcessRegressor(kernel=RBF(1.0, length_scale_bounds=(1e-05, 10000.0)), n_restarts_optimizer=60, random_state=0))

        # TODO: Add custom initialization for your model here if necessary

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean, gp_std = self.gaussian_process.predict(test_features, return_std=True)


        # TODO: Use the GP posterior to form your predictions here
        # gp_mean = np.ones_like(test_features[:,0])
        # gp_std = np.ones_like(test_features[:,0])
        # count=0
        # for x in range(0,2):
        #     for y in range(0,2):
        #         local_gp = self.gp_2dlist[x][y]
        #         bool_array = np.logical_and(np.logical_and(test_features[:,0]>=0.2*x,test_features[:,0]<=0.2*(x+1)),np.logical_and(test_features[:,1]>=0.2*y,test_features[:,1]<=0.2*(y+1)))
        #         inds = np.where(bool_array)[0]
        #         count += len(inds)
        #         print(f"going to predict x={x} y={y} with ds size {len(inds)} (len is {count} so far)")
        #         gp_mean[inds], gp_std[inds] = local_gp.predict(test_features[inds,:], return_std=True)

        # predictions = gp_mean*1.1
        # return (predictions, gp_mean, gp_std)

        #predictions = self.gaussian_process.sample_y(test_features, random_state=0)[:,0]
        predictions = gp_mean

        # print(f"predictions has dimensions {predictions.shape}")
        # print(f"gp_mean has dimensions {gp_mean.shape}")
        # print(f"gp_Std has dimensions {gp_std.shape}")

        return (1.05*predictions, gp_mean, gp_std)

    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        print("entering fitting model")

        # TODO: Fit your model here

        # kmeans = KMeans(n_clusters=7000, random_state=0).fit(train_features)
        # cluster_centers = kmeans.cluster_centers_

        # closest_points_coords = []
        # closest_points_labels = []
        # print("after kmeans")
        # count = 0
        # for cc in cluster_centers:
        #     if count%500==0:
        #         print(f"count is {count}")
        #     deltas = train_features - cc
        #     dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        #     closest_ind = np.argmin(dist_2)
        #     closest_points_coords.append(train_features[closest_ind])
        #     closest_points_labels.append(train_GT[closest_ind])
        #     count+=1
        # print("after finding closest to centers")
        # closest_points_coords = np.array(closest_points_coords)
        # closest_points_labels = np.array(closest_points_labels)

        # self.gaussian_process.fit(closest_points_coords, closest_points_labels)

        # count=0
        # for x in range(0,5):
        #     for y in range(0,5):
        #         local_gp = self.gp_2dlist[x][y]
        #         bool_array = np.logical_and(np.logical_and(train_features[:,0]>=0.2*x,train_features[:,0]<=0.2*(x+1)),np.logical_and(train_features[:,1]>=0.2*y,train_features[:,1]<=0.2*(y+1)))
        #         inds = np.where(bool_array)[0]
        #         count += len(inds)
        #         print(f"going to fit x={x} y={y} with ds size {len(inds)} (len is {count} so far)")
        #         local_gp.fit(train_features[inds,:],train_GT[inds])

        # idx = np.random.randint(15000, size=3000)
        inds = np.random.choice(range(len(train_features)), 3000)
        # sample_train_features = train_features[inds] #train_features[:3000]
        # sample_train_GT = train_GT[inds] #train_GT[:3000]
        sample_train_features = train_features[:3000]
        sample_train_GT = train_GT[:3000]   

        print(f"before fit")
        self.gaussian_process.fit(sample_train_features, sample_train_GT)
        print(f"after fit")

        pass


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    #assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape
    print(f" ground_truth.ndim { ground_truth.ndim}")
    assert ground_truth.ndim == 1
    print(f" predictions.ndim { predictions.ndim}")
    assert predictions.ndim == 1 
    print(f" predictions.shape { predictions.shape}")
    print(f" ground_truth.shape { ground_truth.shape}")
    assert ground_truth.shape == predictions.shape


    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
