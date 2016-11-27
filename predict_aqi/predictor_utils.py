import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

from predict_aqi import config


def train_regressor(clf, x_train, y_train):
    print("Training {}...".format(clf.__class__.__name__))
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    print("Done!\nTraining time (secs): {:.3f}".format(end - start))


def predict_values(clf, features):
    print("Predicting labels using {}...".format(clf.__class__.__name__))
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print("Done!\nPrediction time (secs): {:.3f}".format(end - start))
    return y_pred


def print_mean_absolute_error(y_actual, y_pred):
    print(
        "Mean absolute error for training set (interpolation): {}".format(
            mean_absolute_error(y_actual, y_pred) * config.MAX_AQI
        )
    )