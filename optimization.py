import keras_tuner

def optimizer(model, x_train, y_train, x_test, y_test):
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="BCNN",
    )
    tuner.search(x_train, y_train, epochs = 20, validation_data = (x_test, y_test))

    return
