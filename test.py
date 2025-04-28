from functions import *
from scipy.constants import e as e
from scipy.constants import hbar as hbar
from scipy.constants import k as kb
from scipy.constants import m_e as me
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

def plot_real_space(nx_number, ny_number, a1, a2, delta_1, delta_2):
    for nx_i in np.arange(-int(nx_number / 2),int(nx_number / 2),1) :
        for ny_i in np.arange(-int(ny_number / 2),int(ny_number / 2),1):
            point1 = a1 * nx_i + a2 * ny_i + delta_1
            point2 = a1 * nx_i + a2 * ny_i + delta_2
            plt.scatter(point1[0],point1[1],c='red',s = 1)
            plt.scatter(point2[0],point2[1],c='blue',s = 1)
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_reciprocal_space(a, nx_number, ny_number, b1, b2):
    delta_k1 = b1 / nx_number
    delta_k2 = b2 / ny_number
    plot_lattice_x = np.array([])
    plot_lattice_y = np.array([])
    for nx_i in np.arange(-1, 2, 1):
        for ny_i in np.arange(-1, 2, 1):
            point = b1 * nx_i + b2 * ny_i
            plt.scatter(point[0], point[1], c='blue', s = 10)
            for nx_sub_i in np.arange(0, nx_number, 1):
                for ny_sub_i in np.arange(0, ny_number, 1):
                    point_sub = b1 * nx_i + b2 * ny_i + nx_sub_i * delta_k1 + ny_sub_i * delta_k2
                    plot_lattice_x = np.append(plot_lattice_x,point_sub[0])
                    plot_lattice_y = np.append(plot_lattice_y,point_sub[1])
    plt.scatter(plot_lattice_x, plot_lattice_y, c='red',s=1)
    K = np.array([4 * np.pi / (3 * a),0])
    plt.scatter(K[0],K[1],c='black',s=5)
    print('finish')
    plt.axis('equal')
    plt.grid(True)
    plt.show() 

def generate_sampling_near_k_1d(nx_number, ny_number, b1, b2):
    delta_k1 = b1 / nx_number
    delta_k2 = b2 / ny_number
    k_array_x = np.array([])
    k_array_y = np.array([])
    for nx_sub_i in np.arange(0, nx_number ,1):
        k_x = delta_k1[0] * nx_sub_i + delta_k2[0] * nx_sub_i
        k_y = delta_k1[1] * nx_sub_i + delta_k2[1] * nx_sub_i
        k_array_x = np.append(k_array_x, k_x)
        k_array_y = np.append(k_array_y,k_y)
    k_array = np.vstack((k_array_x,k_array_y))
    return k_array

def plot_band_near_k_1d(nx_number,ny_number,b1,b2,K):
    k_array = generate_sampling_near_k_1d(nx_number,ny_number,b1,b2)
    e_array = np.array([])
    plt.scatter(K[0],K[1],c='black',s=5)
    plt.scatter(k_array[0],k_array[1],c='red',s=3)
    plt.show()

    for kx in k_array[0]:
        q = kx - K[0]
        energy = np.square(hbar) * np.square(q) / (2 * m_eff_c) * energy_norm + energy_shift# in ev
        e_array = np.append(e_array,energy) 
    plt.plot(k_array[0],e_array)
    plt.scatter(K[0],K[1],c='black',s=5)
    plt.grid()
    plt.xlabel('k [A^-1]')
    plt.ylabel('energy [ev]')
    plt.show()

def generate_sampling_near_k_2d(nx_number, ny_number, b1,b2,qm): #qm is for the maximum q value
    delta_k1 = b1 / nx_number
    delta_k2 = b2 / ny_number
    q_array_x = np.array([])
    q_array_y = np.array([])
    for nx_sub_i in np.arange(0, nx_number ,1):
        for ny_sub_i in np.arange(0,ny_number, 1):
            q_x = delta_k1[0] * nx_sub_i + delta_k2[0] * ny_sub_i - K[0]
            q_y = delta_k1[1] * nx_sub_i + delta_k2[1] * ny_sub_i - K[1]
            if (np.square(q_x) + np.square(q_y)) < qm**2:
                q_array_x = np.append(q_array_x, q_x)
                q_array_y = np.append(q_array_y,q_y)
    q_array = np.vstack((q_array_x,q_array_y))
    return q_array

def plot_band_near_k_2d(nx_number, ny_number,b1,b2,energy_max):
    qm = np.sqrt(2 * m_eff_c * energy_max / energy_norm) / hbar# in A
    print(qm)
    q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2,qm)
    plt.scatter(q_array[0],q_array[1],s=1)
    plt.axis('equal')
    plt.grid()
    plt.show()

def calculate_total_number_electron(nx_number, ny_number, b1, b2, qm):
    q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)
    q_x_array = q_array[0]
    q_y_array = q_array[1]
    #plt.scatter(q_array[0],q_array[1],s=1)
    #plt.axis('equal')
    #plt.grid()
    #plt.show()
    
    q_square_array = q_x_array**2 + q_y_array**2
    energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c)
    energy_array = energy_array * energy_norm + energy_shift
    #fig = plt.figure(figsize=(10,7))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(q_x_array, q_y_array, energy_array, c=energy_array, cmap='viridis', s=20)
    #ax.view_init(elev=30, azim=135)  # Adjust view angles
    #plt.show()

    f_dirac = 1 / (np.exp((energy_array - ef) / (kb_ev * T)) + 1)
    #plt.scatter(energy_array,f_dirac)
    #plt.show()
    total_number = 0
    energy_average = 0
    for f_index, f_i in enumerate(f_dirac):
        total_number += f_i
        energy_average += f_i * energy_array[f_index]
    energy_average = energy_average / total_number
    print('total number', total_number)
    print('average energy', energy_average)
    return total_number, energy_average

def plot_transverse():
    nx_number_array = np.arange(10,500,50)
    #nx_number_array = np.array([420])
    #T_array = np.arange(10,300,10)
    T_array = np.array([100])
    
    sigma_t_array = np.array([])
    carrier_density_array = np.array([])
    for nx_number_index, nx_number in enumerate(nx_number_array):
        print(nx_number)
        ny_number = nx_number
        total_number, energy_average = calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)
        area_bz = (2 * np.pi)**2 / area_unit_cell
        carrier_density = total_number  / ((nx_number * ny_number) * area_unit_cell) * 2 * 2
        #carrier_density = total_number / area_unit_cell
        carrier_density_array = np.append(carrier_density_array,carrier_density)

        ny_number = nx_number
        q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)
        q_x_array = q_array[0]
        q_y_array = q_array[1]
        q_y_number = q_y_array.shape[0]
        v_x = hbar**2 * q_x_array / m_eff_c
        v_y = hbar**2 * q_y_array / m_eff_c
        q_square_array = q_x_array**2 + q_y_array**2
        energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c)
        energy_array = energy_array * energy_norm + energy_shift
        sigma_t = 0
        for t in T_array:
            print(t)
            for q_x_index, q_x in enumerate(q_x_array):
                q_y = q_y_array[q_x_index]
                energy_q = (hbar**2 * (q_x**2 + q_y**2)) / (2 * m_eff_c)
                energy_q = energy_q * energy_norm + energy_shift
                tau_ph_t = hbar * rho * np.square(s) / (2 * np.pi * g * np.square(E_c) * kb * t)
                tau_ph_t = hbar**3 * rho * s**2 / (m_eff_c * E_c**2 * kb * t)
                f_dirac_q = 1 / (np.exp((energy_q - ef) / (kb_ev * t)) + 1)
                f_dirac_q_dot = f_dirac_q**2 * np.exp((energy_q - ef) / (kb_ev * t)) / (kb * t)
                v_x_q = hbar * q_x * 1e10 / m_eff_c
                sigma_t += v_x_q**2 * tau_ph_t * f_dirac_q_dot
            sigma_t *= 2 * 2 * e**2 / (area_unit_cell * nx_number * ny_number)
            sigma_t_array = np.append(sigma_t_array,sigma_t)
            sigma_t = 0
    if T_array.shape[0] == 1:
        plt.plot(nx_number_array,sigma_t_array)
        plt.show()
        plt.scatter(nx_number_array, carrier_density_array)
        plt.show()
    else:
        plt.plot(T_array,sigma_t_array)
        plt.show()


def plot_f_dot():
    q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)
    q_x_array = q_array[0]
    q_y_array = q_array[1]    
    plt.scatter(np.arange(q_x_array.shape[0]),q_x_array)
    plt.show()
    q_square_array = q_x_array**2 + q_y_array**2
    energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c)
    energy_array = energy_array * energy_norm + energy_shift
    #fig = plt.figure(figsize=(10,7))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(q_x_array, q_y_array, energy_array, c=energy_array, cmap='viridis', s=20)
    #ax.view_init(elev=30, azim=135)  # Adjust view angles
    #plt.show()

    f_dirac = 1 / (np.exp((energy_array - ef) / (kb_ev * T)) + 1)
    f_dirac_dot = f_dirac**2 * np.exp((energy_array - ef) / (kb_ev * T)) / (kb_ev * T)
    plt.scatter(energy_array,f_dirac)
    plt.scatter(energy_array,f_dirac_dot)
    plt.show()

def plot_scattering_rate_vs_carrier_density_ef_fixed():
    print("T = ", T)
    #print("ef = ", ef)
    nx_number_array = np.arange(100, 500, 100)
    #nx_number_array = np.array([300])
    plt.figure(figsize = (8,6))
    fig_conduc = plt.gcf()
    plt.figure(figsize = (8,6))
    fig_mob = plt.gcf()
    plt.figure(figsize = (8,6))
    fig_sxx = plt.gcf()
    f = True
    for nx_number in nx_number_array:

        q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)
        global ef
        q_x_array = q_array[0]
        q_y_array = q_array[1]
        q_square_array = q_x_array**2 + q_y_array**2
        energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c)
        energy_array = energy_array * energy_norm + energy_shift

        ef_array = np.linspace(-0.1, 0.3, 10)  # in eV
        carrier_density_array = []
        sigma_t_array = []
        mobility_array_cm2 = []
        L_1_array = []
        s_xx_array = []
        scattering_rate = np.array([])
        conductivity_side_jump_array = []

        for ef_i in ef_array:
            ef = ef_i
            sigma_t = 0
            L_1 = 0
            s_xx = 0
            total_number, energy_average = calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)
            area_bz = (2 * np.pi)**2 / area_unit_cell
            carrier_density = total_number  / ((nx_number * ny_number) * area_unit_cell) * 2 * 2
            #carrier_density = total_number / area_unit_cell
            carrier_density_array.append(carrier_density)



            for q_x_index, q_x in enumerate(q_x_array):
                q_y = q_y_array[q_x_index]
                energy_q = (hbar**2 * (q_x**2 + q_y**2)) / (2 * m_eff_c)
                energy_q = energy_q * energy_norm + energy_shift

                tau_ph_t = hbar**3 * rho * s**2 / (m_eff_c * E_c**2 * kb * T)

                f_dirac_q = 1 / (np.exp((energy_q - ef) / (kb_ev * T)) + 1)
                f_dirac_q_dot = f_dirac_q**2 * np.exp((energy_q - ef) / (kb_ev * T)) / (kb * T)
                v_x_q = hbar * q_x * 1e10 / m_eff_c  # Convert Å⁻¹ to m⁻¹

                sigma_t += v_x_q**2 * tau_ph_t * f_dirac_q_dot
                L_1 += v_x_q**2 * tau_ph_t * f_dirac_q_dot * (energy_q - ef) * e #convert energy from ev to j


            sigma_t *= 2 * 2 * e**2 / (area_unit_cell * nx_number * ny_number)
            L_1 *= 4 * e**2 / (area_unit_cell * nx_number * ny_number)
            s_xx =  1 / (e * T) * (L_1 / sigma_t)
            sigma_t_array.append(sigma_t)
            L_1_array.append(L_1)
            s_xx_array.append(s_xx)
            E_gap = energy_gap
            E_gap = E_gap * e #convert from ev to J
            print("enery gap is: ", E_gap)
            gamma_square = hbar**2 * E_gap / (2 * m_eff_c)
            xi = gamma_square / (E_gap**2)
            print("xi is: ", xi)
            conductivity_side_jump = 2 * e**2 / hbar * xi * carrier_density
            mobility_side_jump = conductivity_side_jump / (carrier_density * e) * 1e4 #convert to cm^-2
            print("conductivity of side jump is: ", conductivity_side_jump)
            print("mobility of side jump is in cm: ", mobility_side_jump)
            conductivity_side_jump_array.append(conductivity_side_jump)

            # Mobility: µ = σ / (n e), convert to cm²/Vs
            if carrier_density != 0:
                mobility_cm2 = (sigma_t / (carrier_density * e)) * 1e4
            else:
                mobility_cm2 = 0
            mobility_array_cm2.append(mobility_cm2)

            # Debug
            print("tau_ph_t =", tau_ph_t)
            print("v_x_q =", v_x_q)
            print("carrier_density =", carrier_density)
            print("sigma_t =", sigma_t)
            print("mobility (cm²/Vs) =", mobility_cm2)

        # Convert to numpy arrays
        carrier_density_array = np.array(carrier_density_array)
        sigma_t_array = np.array(sigma_t_array)
        mobility_array_cm2 = np.array(mobility_array_cm2)
        carrier_density_array = carrier_density_array * 1e-4
        L_1_array = np.array(L_1_array)
        s_xx_array = np.array(s_xx_array)

        # Plot conductivity
        plt.figure(fig_conduc.number)
        sigma_t_array = sigma_t_array 
        plt.plot(carrier_density_array, sigma_t_array, marker='o', label=f'nx_number ={nx_number} ')
        plt.plot(carrier_density_array,conductivity_side_jump_array,marker='x',label='side jump')
        # Plot mobility
        plt.figure(fig_mob.number)
        plt.plot(carrier_density_array, mobility_array_cm2, marker='s', color='green', label=f'nx_number ={nx_number} ')
        plt.figure(fig_sxx.number)
        s_xx_array = s_xx_array / 1e-6
        plt.plot(carrier_density_array,s_xx_array)


    plt.figure(fig_conduc.number)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Carrier Density (1/m²)')
    plt.ylabel('Conductivity')
    plt.title(f'Conductivity vs Carrier Density at T = {T} K')
    plt.grid(True)
    plt.legend()


    plt.figure(fig_mob.number)
    plt.xlabel('Carrier Density (1/m²)')
    plt.xscale('log')
    plt.ylabel('Mobility (cm²/Vs)')
    plt.title(f'Mobility vs Carrier Density at T = {T} K')
    plt.grid(True)
    plt.legend()

    plt.figure(fig_sxx.number)
    plt.xscale('log')
    plt.xlim((1e12,1e14))
    plt.show()

    return carrier_density_array, sigma_t_array, mobility_array_cm2

def test_size_converge():
    print("T = ", T)
    print("ef = ", ef)
    nx_number_array = np.arange(100, 500, 50)
    nx_number_array = np.array([300])
    plt.figure(figsize = (8,6))
    fig_conduc = plt.gcf()
    plt.figure(figsize = (8,6))
    fig_mob = plt.gcf()
    plt.figure(figsize = (8,6))
    fig_sxx = plt.gcf()
    f = True
    for nx_number in nx_number_array:
        carrier_density_array = []
        sigma_xx_array = []
        L_1_array = []
        s_xx_array = []
        mobility_array = []
        ny_number = nx_number
        print("nx number is: ",nx_number)
        q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)# This is in A
        q_x_array = q_array[0]
        q_y_array = q_array[1]
        q_square_array = q_x_array**2 + q_y_array**2
        energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c) #This energy is in J
        energy_array = energy_array * energy_norm + energy_shift #This convert energy from J to ev, energy_shift is set to 0
        sigma_xx = 0
        s_xx = 0
        L_1 = 0

        total_number, energy_average = calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)
        carrier_density = total_number / ((nx_number * ny_number) * area_unit_cell) * 2 * 2 #This is in m-2
        carrier_density_array.append(carrier_density)
        for q_x_index, q_x in enumerate(q_x_array):
            q_y = q_y_array[q_x_index]
            energy_q = (hbar**2 * (q_x**2 + q_y**2)) / (2 * m_eff_c) #in J
            energy_q = energy_q * energy_norm + energy_shift # convert from ev to J
            tau_ph_t = hbar**3 * rho * s**2 / (m_eff_c * E_c**2 * kb * T)
            f_dirac_q = 1 / (np.exp((energy_q - ef) / (kb_ev * T)) + 1)
            f_dirac_q_dot = f_dirac_q**2 * np.exp((energy_q - ef) / (kb_ev * T)) / (kb * T) #in J-1
            v_x_q = hbar * q_x * 1e10 / m_eff_c #in m-1
            sigma_xx += v_x_q**2 * tau_ph_t * f_dirac_q_dot
            L_1 += v_x_q**2 * tau_ph_t * f_dirac_q_dot * (energy_q - ef) * e
        sigma_xx *= 2 * 2 * e**2 / (area_unit_cell * nx_number * ny_number)
        L_1 *= 4 * e**2 / (area_unit_cell * nx_number * ny_number)
        s_xx = 1 / (e * T) * (L_1 / sigma_xx)
        L_1_array.append(L_1)
        s_xx_array.append(s_xx)
        sigma_xx_array.append(sigma_xx)
        mobility = (sigma_xx / (carrier_density * e)) * 1e4 #in cm2
        mobility_array.append(mobility)
    
        carrier_density_array = np.array(carrier_density_array) * 1e-4 #in cm-2
        mobility_array = np.array(mobility_array)
        L_1_array = np.array(L_1_array)
        s_xx_array = np.array(s_xx_array)
        plt.figure(fig_conduc.number)
        plt.scatter(carrier_density_array, sigma_xx_array, marker='o', label=f'nx={nx_number}')

    plt.figure(fig_conduc.number)
    plt.show()
        
        


def plot_transverse_nx_number(ef,t, b1, b2):
    print("Temperature is: ", T)
    print("fermi energy is: ", ef)
    nx_number_array = np.arange(100,500,50)
    carrier_density_array = []
    sigma_t_array = []
    mobility_array_cm2 = []
    L_1_array = []
    s_xx_array = []
    for nx_number in nx_number_array:
        ny_number = nx_number
        print("nx number is: ",nx_number)
        print("ny_number is: ",ny_number)
        q_array = generate_sampling_near_k_2d(nx_number, ny_number, b1, b2, qm)
        q_x_array = q_array[0]
        q_y_array = q_array[1]
        q_square_array = q_x_array**2 + q_y_array**2
        energy_array = (hbar**2 * q_square_array) / (2 * m_eff_c)
        energy_array = energy_array * energy_norm + energy_shift

        sigma_t = 0
        L_1 = 0
        s_xx = 0

        total_number, energy_average = calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)
        area_bz = (2 * np.pi)**2 / area_unit_cell
        carrier_density = total_number  / ((nx_number * ny_number) * area_unit_cell) * 2 * 2
        #carrier_density = total_number / area_unit_cell
        carrier_density_array.append(carrier_density)


        for q_x_index, q_x in enumerate(q_x_array):
            q_y = q_y_array[q_x_index]
            energy_q = (hbar**2 * (q_x**2 + q_y**2)) / (2 * m_eff_c)
            energy_q = energy_q * energy_norm + energy_shift

            tau_ph_t = hbar**3 * rho * s**2 / (m_eff_c * E_c**2 * kb * t)

            f_dirac_q = 1 / (np.exp((energy_q - ef) / (kb_ev * t)) + 1)
            f_dirac_q_dot = f_dirac_q**2 * np.exp((energy_q - ef) / (kb_ev * t)) / (kb * t)
            v_x_q = hbar * q_x * 1e10 / m_eff_c  # Convert Å⁻¹ to m⁻¹

            sigma_t += v_x_q**2 * tau_ph_t * f_dirac_q_dot
            L_1 += v_x_q**2 * tau_ph_t * f_dirac_q_dot * (energy_q - ef) * e #convert energy from ev to j


        sigma_t *= 2 * 2 * e**2 / (area_unit_cell * nx_number * ny_number) #in m
        L_1 *= 4 * e**2 / (area_unit_cell * nx_number * ny_number)
        s_xx =  1 / (e * t) * (L_1 / sigma_t)
        sigma_t_array.append(sigma_t)
        L_1_array.append(L_1)
        s_xx_array.append(s_xx)

        if carrier_density != 0:
            mobility_cm2 = (sigma_t / (carrier_density * e)) * 1e4 #in cm^2
        else:
            mobility_cm2 = 0
        mobility_array_cm2.append(mobility_cm2)

    # Convert to numpy arrays
    carrier_density_array = np.array(carrier_density_array)
    sigma_t_array = np.array(sigma_t_array)
    mobility_array_cm2 = np.array(mobility_array_cm2)
    carrier_density_array = carrier_density_array * 1e-4 #in cm^-2
    L_1_array = np.array(L_1_array)
    s_xx_array = np.array(s_xx_array)

    # Plot conductivity
    plt.figure(figsize=(8, 6))
    #sigma_t_array = sigma_t_array *(1/6.5) * 1e10
    plt.plot(nx_number_array, sigma_t_array, marker='o', label='Conductivity (S)')
    plt.yscale('log')
    plt.xlabel('nx_number')
    plt.ylabel('Conductivity')
    plt.title(f'Conductivity vs Carrier Density at T = {t} K')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot mobility
    plt.figure(figsize=(8, 6))
    plt.plot(nx_number_array, mobility_array_cm2, marker='s', color='green', label='Mobility (cm²/Vs)')
    plt.xlabel('nx_number')
    plt.ylabel('Mobility (cm²/Vs)')
    plt.title(f'Mobility vs Carrier Density at T = {t} K')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(nx_number_array,carrier_density_array, marker='s', color='green', label='carrier density')
    plt.xlabel('nx_number')
    plt.ylabel('carrier density cm^-2')
    plt.grid(True)
    plt.show()

    return carrier_density_array, sigma_t_array, mobility_array_cm2


def calculate_side_jump(m_eff,E_gap,nx_number,ny_number):
    E_gap = E_gap * e #convert from ev to J
    print("enery gap is: ", E_gap)
    gamma_square = hbar**2 * E_gap / (2 * m_eff)
    xi = gamma_square / (E_gap**2)
    print("xi is: ", xi)
    total_number, energy_average = calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)
    area_bz = (2 * np.pi)**2 / area_unit_cell
    carrier_density = total_number  / ((nx_number * ny_number) * area_unit_cell) * 2 * 2
    conductivity_side_jump = 2 * e**2 / hbar * xi * carrier_density
    mobility_side_jump = conductivity_side_jump / (carrier_density * e) * 1e4 #convert to cm^-2
    print("conductivity of side jump is: ", conductivity_side_jump)
    print("mobility of side jump is in cm: ", mobility_side_jump)






    
kb_mev = kb / (1e-3 * e)
kb_ev = kb / (e)
m_eff_c = 0.40 * me
energy_gap = 1.6
energy_norm = 1e20 / e # J to ev  e[ev] = e_norm * e[J], also remember to use m in e[J] and A in e[ev]
energy_shift = energy_gap / 2 # half of the band gap
energy_shift = 0
T = 300 #K


a = 3.16 # in A
nx_number = 300
ny_number = 300
a1 = a * np.array([0.5,np.sqrt(3) / 2])
a2 = a * np.array([0.5,-np.sqrt(3) / 2])
area_unit_cell = np.abs(a1[0] * a2[1] - a2[0] * a1[1]) * 1e-20
print(area_unit_cell)
delta_1 = a * np.array([0.5,np.sqrt(3) / 6])
delta_2 = a * np.array([0.5,-np.sqrt(3) / 6])
b1 = 2 * np.sqrt(3) / (3 * a) * np.array([np.sqrt(3) / 2, 1 / 2]) * 2 * np.pi
b2 = 2 * np.sqrt(3) / (3 * a) * np.array([np.sqrt(3) / 2, - 1 / 2]) * 2 * np.pi

K = np.array([4 * np.pi / (3 * a),0]) 


#plot_real_space(nx_number, ny_number, a1, a2, delta_1, delta_2)
#plot_reciprocal_space(a,nx_number, ny_number, b1, b2)
#plot_band_near_k_1d(nx_number,ny_number,b1,b2,K)
energy_max = 0.5 #ev
ef = 0.2
qm = np.sqrt(2 * m_eff_c * (energy_max-energy_shift) / energy_norm) / hbar# in A
#plot_band_near_k_2d(nx_number, ny_number, b1, b2, energy_max)
#calculate_total_number_electron(nx_number, ny_number, b1, b2, qm)


print(times(a1, b1), times(a1, b2), times(a2, b1), times(a2, b2))
total_number, energy_average = calculate_total_number_electron(nx_number,ny_number,b1,b2,qm)
area_bz = (2 * np.pi)**2 / area_unit_cell
carrier_density = total_number * area_bz / (nx_number * ny_number)
print("carrier_density: ", carrier_density)
rho = 3.1 * 1e-7 #g/cm^2
rho = rho * 1e-3 * 1e4 #kg/m^2

s = 2100 #m/s

E_c = 2.4 #ev
E_c = E_c * e

g = m_eff_c / (2 * np.pi * np.square(hbar))
tau_ph = hbar * rho * np.square(s) / (2 * np.pi * g * np.square(E_c) * kb * T)

skew_part = 4 * np.pi * (kb * T / (rho * s**2)) * g * energy_average * e * tau_ph / hbar * e
skew_part2 = 2 * energy_average * e / (E_c**2) * e
E_c_square_test = (hbar * rho * s**2) / (2 * np.pi * g * kb * T * tau_ph) / e
print('tau_ph is', tau_ph)
print(skew_part)
print('skew part is', skew_part2)
print(E_c_square_test)
#plot_transverse()
#plot_f_dot()

#test_size_converge()
plot_scattering_rate_vs_carrier_density_ef_fixed()
ef_array = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])  # in eV
#plot_transverse_nx_number(ef,T,b1,b2)
#calculate_side_jump(m_eff_c, energy_gap,nx_number,ny_number)
