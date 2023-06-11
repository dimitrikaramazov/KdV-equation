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


#Analytical double soliton


u_an = []

for i in range(n) :
    
    u_an.append(c1  / (2 * np.square(np.cosh((math.sqrt(c1) * (x - a1 * L - c1 * t[i]) % L / 2)))) + c2  / (2 * np.square(np.cosh((math.sqrt(c2) * ((x - a2 * L - c2 * t[i]) % L) / 2) ))))




#Two solitons individually (analytical) :
    
u_1 = []
u_2 = []
for i in range(n) :
    u_1.append(c1  / (2 * np.square(np.cosh(math.sqrt(c1) * ((x - a1 * L - c1 * t[i]) % L) / 2 ))))
for i in range(n) :
    u_2.append(c2  / (2 * np.square(np.cosh(math.sqrt(c2) * ((x - a2 * L - c2 * t[i] ) % L) / 2   ))) )
    
#Now, the two solitons individually (numerical) :
    
    
cond_init_1 = c1  / (2 * np.square(np.cosh(math.sqrt(c1) * (x - a1 * L) / 2 )))
cond_init_2 = c2  / (2 * np.square(np.cosh(math.sqrt(c2) * (x - a2 * L) / 2 )))
u_n1 = kdv(cond_init_1, N, dt, n, L)   
u_n2 = kdv(cond_init_2, N, dt, n, L)



#Range of approximation :
    
norm = []
n_t = 112500
time = []
for i in range(n_t):
    
    time.append(i*dt)
    
for i in range(n_t):
    
    norm.append(np.abs(np.linalg.norm(u_an[i] - u[i])))
if True :

    plt.plot(time,norm)
    plt.title('The difference between analytical and numerical (2solitons)')
    plt.xlabel('Time (t)')
    plt.ylabel('norm of the difference')
    plt.show()

    #Now for one soliton :
    norm1 = [] 
    for i in range(n):
    
        norm1.append(np.abs(np.linalg.norm(u_n1[i] - u_1[i])))  
   
    plt.plot(t,norm1)
    plt.title('The difference between analytical and numerical (1soliton)')
    plt.xlabel('Time (t)')
    plt.ylabel('norm of the difference')
    plt.show()

    #All the time-space plots :

    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u_1, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('soliton 1 (numerical)')
    plt.show()


    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u_n1, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('soliton 1 (analytical)')
    plt.show()


    
    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u_2, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('soliton 2 (numerical)')
    plt.show()

    
    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u_n2, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('soliton 2 (analytical)')
    plt.show()

   
    [xx,tt]=np.meshgrid(x,t)
    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('2 solitons (numerical)')
    plt.show()
    [xx,tt]=np.meshgrid(x,t)

    levels = np.linspace(0, c1 / 2, 10)
    plt.contourf(xx,tt, u_an, levels, cmap = cm.jet, extend='both')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('2 solitons(analytical)')
    plt.show()

# plot a few snapshots when they colide
if True:
    k = 50000
    for j in range(k, n, 2500):
        plt.clf()
        plt.plot(x, u[j,:])
        plt.ylim(0, 0.5)
        plt.grid()
        plt.savefig("time" + str(j*dt) + ".png")


# make an animation of the colision
if True:
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

