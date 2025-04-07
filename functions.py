import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath 
import functools  
from matplotlib import cm
import csv
import time
import multiprocessing as mp
from scipy.sparse.linalg import eigsh
import pickle
import os
from matplotlib import colors
import numba as nb
from matplotlib.widgets import Slider, TextBox
from scipy.linalg import ishermitian
#define pauli matrix
import scipy
import psutil
import pyvista as pv
from scipy.signal import argrelextrema
sigma_x = np.array([[0,1],[1,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_0 = np.array([[1,0],[0,1]])

sx = sigma_x
sy = sigma_y
sz = sigma_z
s0 = sigma_0

s1 = sx
s2 = sy
s3 = sz

def hc (matrix):
    L = matrix[0,:].shape[0]
    new_matrix = np.zeros((L,L),dtype = complex)
    for y in np.arange(L):
        for x in np.arange(L):
            new_matrix[x,y] = matrix[y,x].conj()
    return new_matrix

def myprint(number, decimal,string):
    if (string == 'real'):
        print(np.real(np.round(number,decimal)))
    else:
        print(np.round(number, decimal))

def plot_3d_band(k1,k2,eigenvalue):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    k1_shape = k1.shape[0]
    k2_shape = k2.shape[0]
    k1_index = 0
    k2_index = 0
    energy = eigenvalue
    k1,k2 = np.meshgrid(k1,k2)
    for surface in np.arange(eigenvalue[0,0,:].shape[0]):
        surf = ax.plot_surface(k1,k2,energy[:,:,surface],cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    plt.xlabel('k1')
    plt.ylabel('k2')
    plt.show()

def times (A, B):
    return np.matmul(A,B)

def output(matrix, filename, decimal_places):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            formatted_row = [f"{value:.{decimal_places}f}" for value in row]
            writer.writerow(formatted_row)
            
def check_equal(A,B,precision):
    absolute_differences = np.abs(A - B)
    return np.all(absolute_differences <= precision)

def kron(a,b,c):
    return np.kron(np.kron(a,b),c)

def plot_real_band_different_k(k_z_array, energy):
    dim = energy[0,:].shape[0]
    fig = plt.figure()
    #print(dim, k_z_array.shape)
    for dim_i in range(dim):
        #print(k_z_array.shape,energy[:,dim_i].shape)
        plt.plot(k_z_array,energy[:,dim_i],'-k')
    plt.xlabel('kz')
    plt.ylabel('E')
    #plt.ylim(-2,2)
    plt.show()

def plot_real_band_different_k_discreat(k_z_array, energy):
    dim = energy[0,:].shape[0]
    fig = plt.figure()
    #print(dim, k_z_array.shape)
    for dim_i in range(dim):
        #print(k_z_array.shape,energy[:,dim_i].shape)
        plt.scatter(k_z_array,energy[:,dim_i],s = 0.5, c='black')
    plt.xlabel('kz')
    plt.ylabel('E')
    plt.ylim(-2,2)
    plt.show()

def plot_real_band_single_k(energy,name):
    name += ".fig.pickle"
    fig,ax = plt.subplots()
    ax.scatter(np.arange(energy.shape[0]),energy,s=0.5)
    #pickle.dump(fig, open(name, 'wb'))
    plt.show()


def plot_compare_k_real(energy_k,energy_real):
    plt.scatter(np.arange(energy_k.shape[0]),energy_k,s=0.5,label = 'energy_k')
    plt.scatter(np.arange(energy_real.shape[0]),energy_real,s=0.5,label = 'energy_real')
    plt.legend()
    plt.show()

def show_figure(name):
    #name += '.fig.pickle'
    figx = pickle.load(open(name, 'rb')) 
    y = np.zeros(4000)
    #plt.plot(y)
    plt.title(name)
    plt.show() 

def get_project_h(projector, h, delete):
    p = projector
    h_new = times(p,times(h,p))
    h_new = np.delete(h_new,delete,0)
    h_new = np.delete(h_new,delete,1)
    return h_new

def calculate_ldos_3d(energy,evector,L_x_number,L_y_number,lattice_delete,L_z_number = 1):
    index = 0
    plot_x = L_x_number
    plot_y = L_y_number
    plot_z = L_z_number
    mid_x = (plot_x - 1) / 2
    mid_y = (plot_y - 1) / 2
    x_branch = int(L_x_number / 2) 
    ldos_array = evector
    for ldos0 in ldos_array[0, :]:
        ldos_array[:, index] = np.square(np.abs(evector[:, index]))
        index += 1

    if lattice_delete[0] == -1:
        plot_lattice = np.zeros((plot_z * plot_x * plot_y,4))
        conducting_array = np.zeros((2 * plot_x + 4 * plot_y + 2 * plot_z - 12,4))
    else:
        plot_lattice = np.zeros((plot_z * (plot_x * plot_y - lattice_delete.shape[0]),4))
        conducting_array = np.zeros((2 * plot_x + 4 * plot_y + 2 * plot_z - 12,4))
    plot_lattice_branch = np.zeros((plot_y * plot_z,4))
    for energy_index in range(energy.shape[0]):
        ldos = ldos_array[:, energy_index]
        lattice_plot = 0
        lattice_c3 = 0
        lattice_c3_branch = 0
        lattice_c3_conducting = 0
        for z0 in np.arange(L_z_number):
            for y0 in range(plot_y):
                for x0 in range(plot_x):
                    lattice_number =  y0 * plot_x + x0
                    if lattice_number not in lattice_delete:
                        y0_old = y0 - mid_y
                        x0_old = x0 - mid_x
                        theta_old = np.arctan(y0_old / x0_old)
                        theta_new = 4 / 2 * (pi - theta_old)
                        radius = np.sqrt(np.square(y0_old) + np.square(x0_old))
                        x0_new = radius * np.cos(theta_new)
                        y0_new = radius * np.sin(theta_new)
                        if lattice_delete[0] == -1:
                            plot_lattice[lattice_c3,0] = x0_old
                            plot_lattice[lattice_c3,1] = y0_old
                            plot_lattice[lattice_c3,2] = z0
                            if x0_old == x_branch - mid_x:
                                plot_lattice_branch[lattice_c3_branch,0] = x0_old
                                plot_lattice_branch[lattice_c3_branch,1] = y0_old
                                plot_lattice_branch[lattice_c3_branch,2] = z0
                        else:
                            plot_lattice[lattice_c3,0] = x0_new
                            plot_lattice[lattice_c3,1] = y0_new
                            plot_lattice[lattice_c3,2] = z0
                            if x0_old == x_branch - mid_x:
                                plot_lattice_branch[lattice_c3_branch,0] = x0_new
                                plot_lattice_branch[lattice_c3_branch,1] = y0_new
                                plot_lattice_branch[lattice_c3_branch,2] = z0
                                plot_lattice_branch[lattice_c3_branch,3] += np.sum(ldos[lattice_plot * 8:lattice_plot * 8 + 8])
                                lattice_c3_branch += 1
                            if (((y0 == 0 and x0 >= x_branch) or (y0 == L_y_number - 1 and x0 >= x_branch) or (x0 == x_branch) or (x0 == L_x_number - 1)) \
                                and (z0 == 0 or z0 == L_z_number - 1)) or ((x0 == x_branch and (y0 == L_y_number / 2 or y0 == L_y_number / 2 - 1)) and \
                                    z0 > 0 and z0 < L_z_number - 1):
                                conducting_array[lattice_c3_conducting,0] = x0_new
                                conducting_array[lattice_c3_conducting,1] = y0_new
                                conducting_array[lattice_c3_conducting,2] = z0
                                conducting_array[lattice_c3_conducting,3] += np.sum(ldos[lattice_plot * 8:lattice_plot * 8 + 8])
                                lattice_c3_conducting += 1
                        plot_lattice[lattice_c3,3] += np.sum(ldos[lattice_plot * 8:lattice_plot * 8 + 8])  
                        lattice_c3 += 1
                        lattice_plot += 1
    total_number = energy.shape[0]
    conducting_array = conducting_array.real / total_number
    plot_lattice = plot_lattice.real / total_number
    plot_lattice_branch = plot_lattice_branch.real / total_number
    output(conducting_array,'conducting_array.csv',10)
    output(plot_lattice,'plot_lattice.csv',10)
    output(plot_lattice_branch,'plot_lattice_branch.csv',10)

    points = plot_lattice[:,0:3]
    points_branch = plot_lattice_branch[:,0:3]
    mesh = pv.PolyData(points)
    mesh_branch = pv.PolyData(points_branch)
    mesh_conducting = pv.PolyData(conducting_array[:,0:3])
    mesh.point_data['myscalars'] = plot_lattice[:,3]
    mesh_branch.point_data['Branch LDOS'] = plot_lattice_branch[:,3]
    mesh_conducting.point_data['Conducting LDOS'] = conducting_array[:,3]

    mesh.plot(point_size=30, render_points_as_spheres=True,clim=[0,np.max(plot_lattice[:,3]) / 2 ])
    mesh_conducting.plot(point_size=30,render_points_as_spheres=True,clim=[0,np.max(conducting_array[:,3]) /2 ])
    mesh_branch.plot(point_size = 30, render_points_as_spheres=True,clim=[0,np.max(plot_lattice_branch[:,3]) / 2])

def read_csv(path,file_name):
    data = []
    with open('hpc' + '/' + path +'/'+file_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row = [float(value) for value in row]
            data.append(row)
    return np.array(data)

def plot_surface_1d(energy, evector, L_z_number):
    dimension = 8
    ldos_array = evector
    index = 0
    for ldos0 in ldos_array[0, :]:
        ldos_array[:, index] = np.square(np.abs(evector[:, index]))
        index += 1

    plot_lattice = np.zeros((L_z_number,2))
    for energy_index in np.arange(energy.shape[0]):
        plot_lattice_index = 0
        print(energy[energy_index])
        ldos = ldos_array[:,energy_index]
        #plot_lattice = np.zeros((L_z_number,2))
        for z in np.arange(L_z_number):
            plot_lattice[plot_lattice_index,1] += np.sum(ldos[plot_lattice_index * dimension:plot_lattice_index * dimension + dimension]) / 4
            plot_lattice[plot_lattice_index,0] = z
            plot_lattice_index += 1
    plt.plot(plot_lattice[:,0],plot_lattice[:,1],label='E = {:.1f}'.format(energy[energy_index]))
    #plt.legend()
    plt.xlabel('Z')
    plt.ylabel('LDOS')
    plt.show()

def find_lowest_magnitude_energies(kx_array, ky_array, energy_array):
    energy_min = np.zeros((kx_array.shape[0], ky_array.shape[0]))
    energy_max = np.zeros((kx_array.shape[0], ky_array.shape[0]))
    
    for kx_index in range(kx_array.shape[0]):
        for ky_index in range(ky_array.shape[0]):
            energies = energy_array[kx_index, ky_index, :]
            negative_energies = energies[energies < 0]
            positive_energies = energies[energies > 0]
            
            if negative_energies.size > 0:
                energy_min[kx_index, ky_index] = np.max(negative_energies)
            else:
                energy_min[kx_index, ky_index] = 0
            
            if positive_energies.size > 0:
                energy_max[kx_index, ky_index] = np.min(positive_energies)
            else:
                energy_max[kx_index, ky_index] = 0
                
    return energy_min, energy_max

def find_extrema_points(energy_matrix):
    flat_indices_min = np.argpartition(np.abs(energy_matrix.flatten()), 5)[:6]
    smallest_magnitude_indices = np.unravel_index(flat_indices_min, energy_matrix.shape)
    return list(zip(*smallest_magnitude_indices))

def plot_lowest_magnitude_energies(kx_array, ky_array, energy_min, energy_max,name,color):
    kx_mesh, ky_mesh = np.meshgrid(kx_array, ky_array, indexing='ij')
    smallest_points = find_extrema_points(energy_min)
    for idx, (kx_idx, ky_idx) in enumerate(smallest_points):
        plt.plot(kx_mesh[kx_idx, ky_idx], ky_mesh[kx_idx, ky_idx],color ,label=name if idx == 0 else "")
    plt.legend()
    





