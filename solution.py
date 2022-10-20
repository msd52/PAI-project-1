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
        # TODO: Add custom initialization for your model here if necessary
        
        # Load the training dateset and test features for model selection
        train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
        train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
        test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

        # Add list of kernels to test for model selection
        kernels = []
        kernel_names = []

        kernel1 = DotProduct()
        kernels.append(kernel1)
        kernel_names.append('DotProduct()')

        kernel2 = RationalQuadratic(length_scale=1.0, alpha=1.5)
        kernels.append(kernel2)
        kernel_names.append('RationalQuadratic(length_scale=1.0, alpha=1.5)')

        kernel3 = Matern(length_scale=1.0, nu=0.01)
        kernels.append(kernel3)
        kernel_names.append('Matern(length_scale=1.0, nu=0.01)')

        kernel4 = RBF(length_scale=100, length_scale_bounds=(1e-4, 1e2))
        kernels.append(kernel4)
        kernel_names.append('RBF(length_scale=100, length_scale_bounds=(1e-4, 1e2))')

        kernel5 = ConstantKernel()*Matern()
        kernels.append(kernel5)
        kernel_names.append('ConstantKernel()*Matern()')

        kernel6 = ConstantKernel()*RBF()
        kernels.append(kernel6)
        kernel_names.append('ConstantKernel()*RBF()')

        kernel7 = ConstantKernel()*Matern()+ConstantKernel()*RBF()+ConstantKernel()*DotProduct()
        kernels.append(kernel7)
        kernel_names.append('ConstantKernel()*Matern()+ConstantKernel()*RBF()+ConstantKernel()*DotProduct()')

        # Perform model selection and choose the best kernel
        bestKernel = None
        bestValue = -500000
        for idx, kernel_param in enumerate(kernels):
            print('Fitting model' + kernel_names[idx])
            self.gaussian_process = GaussianProcessRegressor(kernel=kernel_param, n_restarts_optimizer=5, normalize_y=True, random_state=0)
            self.fitting_model(train_GT, train_features)

            print(self.gaussian_process.log_marginal_likelihood_value_)
            if model.gaussian_process.log_marginal_likelihood_value_ > bestValue:
                bestKernel = kernel_param
                bestValue = self.gaussian_process.log_marginal_likelihood_value_

        # create the GP with the best kernel that has the highest log_marginal_likelihood_value_
        self.gaussian_process = GaussianProcessRegressor(kernel=bestKernel, n_restarts_optimizer=5, normalize_y=True, random_state=0)


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
        predictions = gp_mean
        return (1.05*predictions, gp_mean, gp_std)

    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        print("entering fitting model")

        # TODO: Fit your model here
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
    print('Fitting model in Main')
    model = Model()
    model.fitting_model(train_GT,train_features) 
    
    # Predict on the test features
    print('Predicting on test features')
    predictions = bestModel.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()