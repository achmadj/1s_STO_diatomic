import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import h5py
from pyscf import gto, scf

def w(alpha, R):
    return alpha * R

def S(r):
    return (1 + r + (r**2 / 3)) * np.exp(-r)

def S_(r):
    return (1 - r + (r**2 / 3)) * np.exp(r)

def J(r):
    return -1 / r + np.exp(-2 * r) * (1 + 1 / r)

def K(r):
    return -np.exp(-r) * (1 + r)

def J_(r):
    term = (1 + (11 / 8) * r + (3 / 4) * r**2 + (1 / 6) * r**3) * np.exp(-2 * r)
    return (1 / r) * (1 - term)

def Ei(w):
    return scp.special.expi(w)

def K_(r):
    f_ = Ei(-4 * r)
    f__ = Ei(-2 * r)
    A = S_(r)
    term1 = (25/8 - (23/4)*r - 3*r**2 - (1/3)*r**3) * np.exp(-2 * r)
    term2 = (0.5772 + np.log(r)) * (S(r)**2)
    term3 = (A**2) * f_
    term4 = 2 * A * S(r) * f__
    try:
        result = (1 / 5) * (term1 + (6 / r) * (term2 + term3 - term4))
        return float(result)
    except:
        return complex((1 / 5) * (term1 + (6 / r) * (term2 + term3 - term4)))

def L(r):
    return (np.exp(-r) * (r + 1/8 + 5/(16*r)) +
            np.exp(-3*r) * (-1/8 - 5/(16*r)))

def KE(alpha, R, c):
    w_val = w(alpha, R)
    numerator = (alpha**2 * (1 - 4*c*K(w_val) - 8*c**2*K(w_val)*S(w_val) -
                             4*c**2*S(w_val)**2 + 2*c**2 - 4*c**3*K(w_val) + c**4))
    denominator = 1 + 4*c*S(w_val) + 4*c**2*S(w_val)**2 + 2*c**2 + 4*c**3*S(w_val) + c**4
    return numerator / denominator

def numerator1(Z, alpha, c, w_val):
    return (2 * Z * (-alpha * (1 + 2*c*S(w_val) + c**2) +
            alpha*K(w_val)*(2*c + 4*c**2*S(w_val) + 2*c**3) +
            alpha*J(w_val)*(c**2 + 2*c**3*S(w_val) + c**4)))

def numerator2(Z, alpha, c, w_val):
    return (2 * Z * (-alpha * (c**2 + 2*c**3*S(w_val) + c**4) +
            alpha*K(w_val)*(2*c + 4*c**2*S(w_val) + 2*c**3) +
            alpha*J(w_val)*(1 + 2*c*S(w_val) + c**2)))

def PE(alpha, R, c, Z1, Z2):
    w_val = w(alpha, R)
    num1 = numerator1(Z1, alpha, c, w_val)
    num2 = numerator2(Z2, alpha, c, w_val)
    denum1 = 1 + 4*c*S(w_val) + 4*c**2*S(w_val)**2 + 2*c**2 + 4*c**3*S(w_val) + c**4
    num3 = alpha * (5/8 * (1 + c**4) + 4 * L(w_val) * (c + c**3) +
                     2 * c**2 * J_(w_val) + 4 * c**2 * K_(w_val))
    return (num1 + num2 + num3) / denum1 + Z1 * Z2 / R

def NE(alpha, R, c, Z1, Z2):
    w_val = w(alpha, R)
    num1 = numerator1(Z1, alpha, c, w_val)
    num2 = numerator2(Z2, alpha, c, w_val)
    denum1 = 1 + 4*c*S(w_val) + 4*c**2*S(w_val)**2 + 2*c**2 + 4*c**3*S(w_val) + c**4
    return (num1 + num2) / denum1

def ee(alpha, R, c, Z1, Z2):
    w_val = w(alpha, R)
    denum1 = 1 + 4*c*S(w_val) + 4*c**2*S(w_val)**2 + 2*c**2 + 4*c**3*S(w_val) + c**4
    num3 = alpha * (5/8 * (1 + c**4) + 4 * L(w_val) * (c + c**3) +
                     2 * c**2 * J_(w_val) + 4 * c**2 * K_(w_val))
    return num3 / denum1

def NN(alpha, R, c, Z1, Z2):
    return Z1 * Z2 / R

def heh_1s_energy(alpha, r, c):
    return KE(alpha, r, c) + PE(alpha, r, c, Z1=2, Z2=1)

def h2_1s_energy(alpha, r, c):
    return KE(alpha, r, c) + PE(alpha, r, c, Z1=1, Z2=1)

def hehe_1s_energy(alpha, r, c):
    return KE(alpha, r, c) + PE(alpha, r, c, Z1=2, Z2=2)

def objective(params, R, species, species_Z):
    alpha, c = params
    Z1, Z2 = species_Z[species]
    if species == 'HeH+':
        return heh_1s_energy(alpha, R, c)
    elif species == 'H2':
        return h2_1s_energy(alpha, R, c)
    elif species == 'HeHe':
        return hehe_1s_energy(alpha, R, c)
    else:
        raise ValueError("Unsupported species")

def main():
    """
    Main function to perform energy optimizations and save results to an HDF5 file.
    """
    # Mapping species to their respective atomic numbers (Z1, Z2)
    species_Z = {
        'HeH+': (2, 1),
        'H2': (1, 1),
        'HeHe': (2, 2)
    }

    # List of molecular species to consider
    species_list = ['HeH+', 'H2', 'HeHe']

    # Create an HDF5 file to store molecular data
    with h5py.File('molecular_data.h5', 'w') as f:
        for species in species_list:
            # Retrieve atomic numbers for the species
            Z1, Z2 = species_Z[species]

            # Create a group for each species
            grp = f.create_group(species)

            # Initialize datasets for bond lengths, alpha, c, and energies
            datasets = {
                'bond_lengths': grp.create_dataset('bond_lengths', (0,), maxshape=(None,), dtype='f'),
                'alphas': grp.create_dataset('alphas', (0,), maxshape=(None,), dtype='f'),
                'cs': grp.create_dataset('cs', (0,), maxshape=(None,), dtype='f'),
                'energies': grp.create_dataset('energies', (0,), maxshape=(None,), dtype='f'),
                'ke_energies': grp.create_dataset('ke_energies', (0,), maxshape=(None,), dtype='f'),
                'ne_energies': grp.create_dataset('ne_energies', (0,), maxshape=(None,), dtype='f'),
                'ee_energies': grp.create_dataset('ee_energies', (0,), maxshape=(None,), dtype='f'),
                'nn_energies': grp.create_dataset('nn_energies', (0,), maxshape=(None,), dtype='f')
            }

            # Define bond lengths to iterate over (e.g., from 0.5 to 5.0 Å)
            bond_length_values = np.arange(0.5, 5.01, 0.01)  # Adjust the step as needed

            for R in bond_length_values:
                # Initial guess for alpha and c
                initial_guess = [1.0, 0.1]
                
                # Bounds for alpha and c
                bounds = [(0.1, 5.0), (0.01, 1.0)]
                
                # Perform optimization using L-BFGS-B algorithm
                result = fmin_l_bfgs_b(
                    func=objective,
                    x0=initial_guess,
                    bounds=bounds,
                    approx_grad=True,
                    args=(R, species, species_Z)
                )
                alpha_opt, c_opt = result[0]
                
                # Calculate optimized energy components
                ke_energy = KE(alpha_opt, R, c_opt)
                ne_energy = NE(alpha_opt, R, c_opt, Z1, Z2)
                ee_energy = ee(alpha_opt, R, c_opt, Z1, Z2)
                nn_energy = NN(alpha_opt, R, c_opt, Z1, Z2)
                energy_opt = ke_energy + ne_energy + ee_energy + nn_energy
                # print(energy_opt)      
                
                # Append results to datasets
                for key, dataset in datasets.items():
                    dataset.resize((dataset.shape[0] + 1,))
                
                datasets['bond_lengths'][-1] = R
                datasets['alphas'][-1] = alpha_opt
                datasets['cs'][-1] = c_opt
                datasets['energies'][-1] = energy_opt
                datasets['ke_energies'][-1] = ke_energy
                datasets['ne_energies'][-1] = ne_energy
                datasets['ee_energies'][-1] = ee_energy
                datasets['nn_energies'][-1] = nn_energy
                
                # Optional: Print detailed results (commented out for performance)
                # Uncomment the lines below if you wish to see the progress
                # print(f"Species: {species}, Bond length: {R}, Optimized alpha: {alpha_opt}, "
                #       f"Optimized c: {c_opt}, Ground state energy: {energy_opt}")
                
                # print(f"""
                # Species: {species},
                # Bond length: {R:.2f} Å,
                # Optimized alpha: {alpha_opt:.4f},
                # Optimized c: {c_opt:.4f},
                # Kinetic Energy (KE): {ke_energy:.6f},
                # Nuclear-Electron Energy (NE): {ne_energy:.6f},
                # Electron-Electron Energy (EE): {ee_energy:.6f},
                # Nuclear-Nuclear Energy (NN): {nn_energy:.6f},
                # Ground State Energy: {energy_opt:.6f}
                # """)

    print("Data saved to molecular_data.h5")