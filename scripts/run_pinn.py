"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm_train, T_sensor_data_train = generate_training_data(cfg)
    print("Solving heat equation with physics informed neural network...")
    pinn_params, losses = train_pinn(T_sensor_data_train, cfg)
    T_pred = predict_grid(pinn_params["nn"], x, y, t, cfg)
    
    print("\nGenerating PINN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/pinn/pinn_snapshots.png", show_interactively=True    
        )
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif"
    )

    plt.plot(losses["total"], label="Loss total")
    plt.plot(losses["data"], label="Loss data")
    plt.plot(losses["ic"], label="Loss ic")
    plt.plot(losses["physics"], label="Loss physics")
    plt.plot(losses["bc"], label="Loss bc")
    plt.xlabel("epoch nr.")
    plt.ylabel(r"$\mathcal{L}$")
    plt.title("Plot over losses")
    plt.legend()
    plt.savefig("output/pinn/pinn_losses",  dpi=300, bbox_inches="tight")
    plt.grid()
    plt.show()
    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
