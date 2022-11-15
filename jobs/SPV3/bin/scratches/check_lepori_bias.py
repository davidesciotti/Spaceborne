import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0, 4, 100)
lepori = 0.5125 + 1.377 * z + 0.222 * z ** 2 - 0.249 * z ** 3

plt.figure()
nztab = np.genfromtxt(
    '/vincenzo/SPV3_07_2022/Flagship_1/InputNz/Sources/Flagship/ngbTab-EP10.dat')

lepori_paper = np.asarray([(0.14, 0.758, 0.624, 0.023),
                          (0.26, 2.607, 0.921, 0.135),
                          (0.39, 4.117, 1.116, 0.248),
                          (0.53, 3.837, 1.350, 0.253),
                          (0.69, 3.861, 1.539, 0.227),
                          (0.84, 3.730, 1.597, 0.280),
                          (1.0, 3.000, 1.836, 0.392),
                          (1.14, 2.827, 1.854, 0.481),
                          (1.3, 1.800, 2.096, 0.603),
                          (1.44, 1.078, 2.270, 0.787),
                          (1.62, 0.522, 2.481, 1.057),
                          (1.78, 0.360, 2.193, 1.138),
                          (1.91, 0.251, 2.160, 1.094)])


zbin_centers = nztab[3, :]
bias_maybe = nztab[1, :]
plt.plot(z, lepori, label='Lepori fitting formula')
plt.plot(lepori_paper[:, 0], lepori_paper[:, 2], label='Lepori paper table', marker='o')
plt.plot(zbin_centers, bias_maybe, label='vincenzo ngbTab_EP10', marker='o')
plt.legend()
plt.xlabel('z')
plt.ylabel('bias')
plt.grid()

