import os
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import numpy as np
import pyfftw
import timeit
import time
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift

# Check for image file
if not os.path.exists("ulf7.BMP"):
    raise FileNotFoundError("The file 'ulf7.BMP' was not found.")

# Read the hologram image file
hologram = mpi.imread("ulf7.BMP")
hologram = hologram.astype(float)
plt.figure(1)
plt.title("Hologram")
plt.imshow(hologram, cmap="viridis")

# DC term suppression
hologram2 = hologram - np.mean(hologram)
plt.figure(2)
plt.title("Hologram with DC suppression")
plt.imshow(hologram2, cmap="viridis")

# Hologram dimensions
Nr0, Nc0 = np.shape(hologram)
print("Number of rows and columns in the hologram are:", Nr0, Nc0)

# Physical parameters
wavelength = 632.8e-9  # HeNe laser wavelength (m)
dx = 6.8e-6  # sensor pixel size (m)
d = 1.054  # object distance (m)
m = 1/14
d2 = d * m

k = 2 * np.pi / wavelength

# Coordinate and frequency grids
y_axis = np.linspace(0, Nr0 - 1, Nr0) - Nr0 / 2
x_axis = np.linspace(0, Nc0 - 1, Nc0) - Nc0 / 2
Fr = np.linspace(-0.5, 0.5 - 1/Nr0, Nr0)
Fc = np.linspace(-0.5, 0.5 - 1/Nc0, Nc0)

x, y = np.meshgrid(x_axis, y_axis)
fx, fy = np.meshgrid(Fc, Fr)
x *= dx
y *= dx
fx /= dx
fy /= dx

# Angular Spectrum Method using pyFFTW
def calculate_asm_parameters():
    f = 1 / (1/d + 1/d2)
    L = np.exp(1j * np.pi / (f * wavelength) * (x**2 + y**2))
    z = d2
    argument = k**2 - 4 * np.pi**2 * (fx**2 + fy**2)
    alpha = np.where(argument >= 0, np.sqrt(argument), 0)
    G = np.exp(-1j * alpha * z)
    return L, G

asm_time = timeit.timeit(calculate_asm_parameters, number=100)
print(f"Average ASM parameter calculation time: {asm_time/100:.6f} seconds")
L, G = calculate_asm_parameters()

def angular_spectrum_method():
    field = hologram * L
    fft_field = pyfftw.interfaces.numpy_fft.fft2(field, threads=os.cpu_count())
    return pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifft2(fft_field * G, threads=os.cpu_count()))

# Convolution Method using pyFFTW
def compute_convolution_parameters():
    f = 1/(1/d+1/d2) 
    L = np.exp(1j*np.pi/(f*wavelength)*(np.multiply(x, x) + np.multiply(y, y)))
    rho = np.sqrt(d2**2 + np.multiply(x, x) + np.multiply(y, y))
    g = 1j / wavelength * np.exp(-1j * 2 * np.pi / wavelength * rho) / (rho)
    g1 = pyfftw.interfaces.numpy_fft.fft2(g, threads=os.cpu_count())
    return L, g1

time_convolution_params = timeit.timeit(compute_convolution_parameters, number=100)
L, g1 = compute_convolution_parameters()
print(f"Average iteration time for convolution parameters computation: {time_convolution_params / 100:.6f} seconds")

def convolution_method():
    field = hologram * L
    fft_field = pyfftw.interfaces.numpy_fft.fft2(field, threads=os.cpu_count())
    return pyfftw.interfaces.numpy_fft.ifftshift(pyfftw.interfaces.numpy_fft.ifft2(fft_field * g1, threads=os.cpu_count()))

# Timing both methods
time_angular = timeit.timeit(angular_spectrum_method, number=100)
reconstructed_field1 = angular_spectrum_method()
print(f"Average iteration time for angular spectrum method: {time_angular / 100:.6f} seconds")

time_convolution = timeit.timeit(convolution_method, number=100)
reconstructed_field3 = convolution_method()
print(f"Average iteration time for convolution method: {time_convolution / 100:.6f} seconds")

# Normalize intensity
I1 = np.abs(reconstructed_field1) / np.max(np.abs(reconstructed_field1))
I3 = np.abs(reconstructed_field3) / np.max(np.abs(reconstructed_field3))

# Plotting
plt.figure(3)
plt.title("Reconstruction with angular spectrum")
plt.imshow(I1, cmap="hot", clim=(0.0, 0.4))
plt.colorbar()

plt.figure(4)
plt.title("Reconstruction with convolution")
plt.imshow(I3, cmap="hot", clim=(0.0, 0.4))
plt.colorbar()

# Save reconstructions
mpi.imsave('Angular_spectrum_reconstruction.png', I1, cmap="hot", vmin=0.0, vmax=0.3)
mpi.imsave('Convolution_reconstruction.png', I3, cmap="hot", vmin=0.0, vmax=0.3)

plt.show()