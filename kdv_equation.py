import numpy as np
import scipy.fft as fft
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
# constants
N = 256
n = 500000
dt = 0.0004
L = 50
c1 = 0.75
c2 = 0.4
a1 = 0.33
a2 = 0.65

def kdv(cond_init, N, dt, n, L):
    """Korteweg-de Vries equation using the split-step method in
    periodic boundary conditions"""
    u = np.zeros((n,N))
    u[0,:] = cond_init
    
    K = np.zeros(N)
    K[0] = -N/2
    for i in range(N-1):
        K[i+1] = K[i] +1
    
    for i in range(1,n):
        u_k = (1/N)*fft.fftshift(fft.fft(u[i-1,:]))

        g_k = np.exp(1j * np.power(2*np.pi * K / L, 3) * dt) * u_k

        g = N*fft.ifft(fft.ifftshift(g_k))

        dg_2 = fft.ifft(fft.ifftshift(1j * (2*np.pi/L) * K * fft.fftshift(fft.fft(np.square(g)))))

        u[i,:] = g - 3 * dg_2 * dt
    
    return u

x = np.zeros(N)
for i in range(N):
    x[i] = L *i / N

t = np.zeros(n)
for i in range(n):
    t[i] = i* dt


cond_init = c1  / (2 * np.square(np.cosh(math.sqrt(c1) * (x - a1 * L) / 2))) + c2  / (2 * np.square(np.cosh(math.sqrt(c2) * (x - a2 * L) / 2)))

u = kdv(cond_init, N, dt, n, L)

# plot the evolution of 2 solitons
if True:
    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Korteweg-de Vries equation - 2 solitons')
    plt.savefig('2solitons.png')

# plot a few snapshots when they colide
if False:
    k = 50000
    for j in range(k, n, 2500):
        plt.clf()
        plt.plot(x, u[j,:])
        plt.ylim(0, 0.5)
        plt.grid()
        plt.savefig("time" + str(j*dt) + ".png")


# make an animation of the colision
if False:
    k = 50000
    interval = 100  

    fig, ax = plt.subplots()
    ax.set_ylim(0, 0.5)
    ax.grid()

    line, = ax.plot([], [])

    def update(frame):
        j = k + frame * 2500
        ax.cla()  
        ax.plot(x, u[j, :])
        ax.set_ylim(0, 0.5)
        ax.grid()
        ax.set_title('Time: {:.2f}'.format(j * dt))

    animation = FuncAnimation(fig, update, frames=40, interval=interval)

    animation.save('2_solitons_colision.gif', writer='pillow')
