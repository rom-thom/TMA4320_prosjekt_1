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

    print(f"Expected physics params: alpha = {cfg.alpha}, k = {cfg.k}, h = {cfg.h}, effektkonst = {cfg.source_strength}")

    print(
        "PINN physics params: "
        f"alpha = {float(jnp.exp(pinn_params['log_alpha'][0])):.4f}, "
        f"k = {float(jnp.exp(pinn_params['log_k'])):.4f}, "
        f"h = {float(jnp.exp(pinn_params['log_h'])):.4f}, "
        f"effektkonst = {float(jnp.exp(pinn_params['log_power'])):.4f}"
    )

    T_pred = predict_grid(pinn_params["nn"], x, y, t, cfg)
    
    print("\nGenerating PINN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/pinn/pinn_snapshots.png"    
        )
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/pinn/pinn_animation.gif"
    )

    plt.plot(losses["total"][20:], label="Loss total")
    plt.plot(losses["data"][20:], label="Loss data")
    plt.plot(losses["ic"][20:], label="Loss ic")
    plt.plot(losses["physics"][20:], label="Loss physics")
    plt.plot(losses["bc"][20:], label="Loss bc")
    plt.xlabel("epoch nr.")
    plt.ylabel(r"$\mathcal{L}$")
    plt.title("Plot over losses")
    plt.legend()
    plt.grid()
    plt.savefig("output/pinn/pinn_losses",  dpi=300, bbox_inches="tight")
    plt.show()
    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
