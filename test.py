from functions import *
from scipy.constants import e as e
from scipy.constants import hbar as hbar
from scipy.constants import k as kb

kb_mev = kb / (1e-3 * e)
a = 3.14 # in A


def compare_phonon_number(T):
    energy = np.arange(0.1, 20, 0.1)  # avoid division by zero
    
    # First plot: for the input temperature T
    p_number_exact = 2 / (np.exp(energy / (kb_mev * T)) - 1) + 1
    p_number_appro = 2 * kb_mev * T / energy
    p_number_error = p_number_exact - p_number_appro

    plt.figure(figsize=(8,6))
    plt.plot(energy, p_number_exact, label='Exact phonon number')
    plt.plot(energy, p_number_appro, label='Approximate phonon number')
    plt.plot(energy, p_number_error, label='Error (Exact - Approx)')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Phonon number')
    plt.title(f'Phonon number comparison at T={T} K')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Second plot: percent error for multiple temperatures
    temperatures = np.linspace(0.1, 300, 10)  # 6 temperatures from ~0 to 300 K

    plt.figure(figsize=(8,6))

    for temp in temperatures:
        p_number_exact_temp = 2 / (np.exp(energy / (kb_mev * temp)) - 1) + 1
        p_number_appro_temp = 2 * kb_mev * temp / energy
        p_number_error_temp = p_number_exact_temp - p_number_appro_temp
        p_number_error_percent_temp = p_number_error_temp / p_number_exact_temp * 100

        plt.plot(energy, p_number_error_percent_temp, label=f'T = {temp:.1f} K')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Error (%)')
    plt.title('Percent error in phonon number for different temperatures')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_lattice(nx_number, ny_number, a1, a2, delta_1, delta_2):
    for nx_i in np.arange(-int(nx_number / 2),int(nx_number / 2),1) :
        for ny_i in np.arange(-int(ny_number / 2),int(ny_number / 2),1):
            point1 = a1 * nx_i + a2 * ny_i + delta_1
            point2 = a1 * nx_i + a2 * ny_i + delta_2
            plt.scatter(point1[0],point1[1],c='red',s = 1)
            plt.scatter(point2[0],point2[1],c='blue',s = 1)
    plt.axis('equal')
    plt.grid(True)
    plt.show()








nx_number = 10
ny_number = 10
a1 = a * np.array([0.5,np.sqrt(3) / 2])
a2 = a * np.array([0.5,-np.sqrt(3) / 2])
delta_1 = a * np.array([0.5,np.sqrt(3) / 6])
delta_2 = a * np.array([0.5,-np.sqrt(3) / 6])

b1 = 2 * np.sqrt(3) / (3 * a) * np.array([np.sqrt(3) / 2, 1 / 2]) * 2 * np.pi
b2 = 2 * np.sqrt(3) / (3 * a) * np.array([np.sqrt(3) / 2, - 1 / 2]) * 2 * np.pi

print(times(a1, b1), times(a1, b2), times(a2, b1), times(a2, b2))
