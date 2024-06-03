import numpy as np
import os
import json


def generate_dynamic_noise(path, distributions=["Gamma", "Exponential", "Uniform"], noise_size=1):

    directory=os.path.dirname(__file__)
    file_path = os.path.join(directory, 'lmo_noise_parameters', path)

    with open(file_path, 'r') as file:
        params = json.load(file)

    b = generate_lmo_b(params, distributions=distributions, noise_size=noise_size)
    noise = np.random.laplace(0, 1/b)
    return noise


def generate_lmo_noise(lmo, distributions, noise_size):
    lmo_bs = generate_lmo_b(lmo, distributions=distributions, noise_size=noise_size)
    return np.random.laplace(0, lmo_bs)


def generate_lmo_b(lmo, distributions, noise_size=1):
    us = 0
    
    if "Gamma" in distributions:
        us = us + lmo['a1']*np.random.gamma(lmo["G_k"], lmo["G_theta"], noise_size)
    else:
        us = us + 0
    
    if "Exponential" in distributions:
        us = us + lmo['a3']*np.random.exponential(lmo["E_lambda"], noise_size)
    else:
        us = us + 0
    
    if "Uniform" in distributions:
        us = us + lmo['a4'] * np.random.uniform(lmo["U_a"], lmo["U_b"], noise_size)
    else:
        us = us + 0
    
    return 1/us
