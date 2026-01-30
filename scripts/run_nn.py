"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    x, y, t, T_fdm_train, T_sensor_data_train = generate_training_data(cfg)
    print("Solving heat equation with neural network...")
    nn_params, losses = train_nn(T_sensor_data_train, cfg)
    T_pred = predict_grid(nn_params, x, y, t, cfg)
    
    print("\nGenerating FDM visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/nn/nn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/nn/nn_animation.gif"
    )
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
