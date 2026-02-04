"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)     #sender initielle matriseverdier
    adam_state = init_adam(nn_params)  

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################


    # Nokre tilfeldige startspunkt (initialbetingelse)
    # ic_points = jnp.array([jnp.linspace(cfg.x_min, cfg.x_max, 30), jnp.linspace(cfg.y_min, cfg.y_max, 30)])
    

    def objektiv(current_nn_params, ic_points):
        return cfg.lambda_data * data_loss(current_nn_params, sensor_data, cfg) + cfg.lambda_ic * ic_loss(current_nn_params, ic_points, cfg)
    
    
    current_nn_params = nn_params
    current_state = adam_state

    import numpy as np
    loss_all = np.zeros((cfg.num_epochs, 3)) # [tot, data, ic]
    
    from tqdm import tqdm
    for epoc in tqdm(range(cfg.num_epochs), desc="Training NN"):
        ic_points, key = sample_ic(key, cfg)
        loss_tot, grad_objektiv = jax.value_and_grad(objektiv, 0)(current_nn_params, ic_points)
        loss_data = data_loss(current_nn_params, sensor_data, cfg)
        loss_ic = ic_loss(current_nn_params, ic_points, cfg)
        loss_all[epoc] = (jnp.array([loss_tot, loss_data, loss_ic])) # for å debugge enklare (og oppgåva sa de)
            
        current_nn_params, current_state = adam_step(current_nn_params, grad_objektiv, current_state, lr=cfg.learning_rate)

    nn_params = current_nn_params
    losses["total"] = list(loss_all[:, 0])
    losses["data"] = list(loss_all[:, 1])
    losses["ic"] = list(loss_all[:, 2])


    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)     

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################


    def objektiv(current_pinn_params, ic_epoch, bc_epoch, interior_epoch):   #loss-funksjoner
        L_data = cfg.lambda_data * data_loss(current_pinn_params["nn"], sensor_data, cfg)
        L_ic = cfg.lambda_ic * ic_loss(current_pinn_params["nn"], ic_epoch, cfg)
        L_bc = cfg.lambda_bc * bc_loss(current_pinn_params, bc_epoch, cfg)
        L_ph = cfg.lambda_physics * physics_loss(current_pinn_params, interior_epoch, cfg)
        return L_data + L_bc + L_ic + L_ph
    
    
    current_pinn_params = pinn_params  #initialverdier
    current_state = opt_state  

    import numpy as np
    loss_all_list = []
    loss_all = np.zeros((cfg.num_epochs, 5)) # [tot, data, ic, bc, physics]
    
    from tqdm import tqdm
    for epoc in tqdm(range(cfg.num_epochs), desc="Training PINN"):

        # Nyt sample kvar iterasjon
        interior_epoch, key = sample_interior(key, cfg)  
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)


        # val_grad = jax.jit(jax.value_and_grad(objektiv, argnums=0))
        # loss_tot, grad_objektiv = val_grad(current_pinn_params, ic_epoch, bc_epoch, interior_epoch)

        loss_tot, grad_objektiv = jax.value_and_grad(objektiv, 0)(current_pinn_params, ic_epoch, bc_epoch, interior_epoch) #deriverer mhp current_pinn_params
        loss_data = data_loss(current_pinn_params["nn"], sensor_data, cfg) 
        loss_ic = ic_loss(current_pinn_params["nn"], ic_epoch, cfg)
        loss_bc = bc_loss(current_pinn_params, bc_epoch, cfg)
        loss_ph = physics_loss(current_pinn_params, interior_epoch, cfg)
        loss_all_list.append([loss_tot, loss_data, loss_ic, loss_bc, loss_ph])  #legger til lossene i en 2d-liste
        loss_all[epoc] = (np.array([loss_tot, loss_data, loss_ic, loss_bc, loss_ph])) # for å debugge enklare
            
        current_pinn_params, current_state = adam_step(current_pinn_params, grad_objektiv, current_state, lr=cfg.learning_rate)

    losses["total"] = [row[0] for row in loss_all_list]
    losses["data"] = [row[1] for row in loss_all_list]
    losses["ic"] = [row[2] for row in loss_all_list]
    losses["bc"] = [row[3] for row in loss_all_list]
    losses["physics"] = [row[4] for row in loss_all_list]

    pinn_params = current_pinn_params

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
