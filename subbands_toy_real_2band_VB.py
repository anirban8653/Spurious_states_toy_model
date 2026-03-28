import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def H_simple(kx, ky, kz):
    """Hamiltonian of a two band mdoel VB-VB"""
    h01 = 13.4106 *(kx - 1j* ky) *kz
    h10 = np.conjugate(h01)
    H_matrix = np.array([
        [   -18.3007*(kx**2 + ky**2) - 7.22796*kz**2, h01],
        [h10,    -0.3622 - 7.88749 * (kx**2 + ky**2) - 16.5208 * kz**2]
    ])
        
    
    return H_matrix


Eg = 0.632

kx = 0.0
# k-space grid
ky_vals = np.linspace(-0.2, 0.2, 100)
kz_vals = np.linspace(-0.2, 0.2, 100)

KY, KZ = np.meshgrid(ky_vals, kz_vals)

# Store eigenvalues
E1 = np.zeros_like(KY)
E2 = np.zeros_like(KY)


# Diagonalisation loop
for i in range(KY.shape[0]):
    for j in range(KY.shape[1]):
        ky = KY[i, j]
        kz = KZ[i, j]
        
        H = H_simple(kx, ky, kz)
        eigvals = eigh(H, eigvals_only=True)
        
        
        E1[i, j] = eigvals[0]  # lower band
        E2[i, j] = eigvals[1]  # upper band


# -------------------------
# 3D Plot
# -------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(KY, KZ, E1)
ax.plot_surface(KY, KZ, E2)

ax.set_xlabel('ky')
ax.set_ylabel('kz')
ax.set_zlabel('Energy')

ax.set_title('3D Dispersion at kx = 0')

plt.show()


#%%

#-------------------------------------
# sine wave expansion
#------------------------------------

def idx(alpha, m, n):
    return alpha + nband * (m + Ny * n)

def ky(m, mp, L):
    """<m,n|ky|m',n'>"""
    m, mp = m + 1, mp + 1
    if (m + mp) % 2 == 1:
        return (-4j / L) * (m * mp) / (m**2 - mp**2)
    return 0

def kz(n, np_, L):
    """<m,n|kz|m',n'>"""
    n, np_ = n + 1, np_ + 1
    if (n + np_) % 2 == 1:
        return (-4j / L) * (n * np_) / (n**2 - np_**2)
    return 0

def kykz(m,n,mp,np_,L):
    """<m,n|kykz|m',n'>"""
    m,mp, n, np_ = m + 1, mp + 1, n + 1, np_ + 1
    if (m + mp) % 2 == 1 and  (n + np_) % 2 == 1:
        return -16 * m * mp * n * np_/ (L**2 * (m**2 - mp**2) * (n**2 - np_**2))
    return 0

def ky2(m,mp,L):
    """<m,n|ky^2|m',n'>"""
    m+=1 
    mp+=1
    if m==mp:
        return (m*np.pi/L)**2
    return 0

def kz2(n,np_,L):
    """<m,n|kz^2|m',n'>"""
    n+=1 
    np_+=1
    if n==np_:
        return (n*np.pi/L)**2
    return 0


def H_k_only(kx, ky, kz, kykz, ky2, kz2):
    """hamiltonian as a function of new variables <m,n||m'n'>"""
    h01 = 13.4106 *(kx*kz - 1j* kykz) 
    h10 = np.conjugate(h01)
    return np.array([
        [   -18.3007*(kx**2 + ky2) - 7.22796*kz2, h01],
        [h10,     - 7.88749 * (kx**2 + ky2) - 16.5208 * kz2]
    ])



#--------------------
# Parameters
#--------------------

L_values = [100,300, 500, 1000, 5000, 10000]

for L in L_values:
    Ny = Nz = 20 
    nband = 2
    Eg = 0.632 #energy gap
    
    dim = nband * Ny * Nz
    H = np.zeros((dim, dim), dtype=complex)
    
    
    #-----------------------------------
    # Constructing the Hamiltonian
    #-----------------------------------
    
    for m in range(Ny):
        for n in range(Nz):
            for mp in range(Ny):
                for np_ in range(Nz):
                    
                    # 1. Calculate spatial elements
                    ky_v = ky(m, mp, L)
                    kz_v = kz(n, np_, L)
                    ky2_v = ky2(m, mp, L)
                    kz2_v = kz2(n, np_, L)
                    kykz_v = kykz(m, n, mp, np_, L)
                    
                    # 2. Get ONLY the k-dependent part (No Eg here)
                    H_k_part = H_k_only(0, ky_v, kz_v, kykz_v, ky2_v, kz2_v)
                    
                    # 3. Add the constants ONLY on the spatial diagonal (m=mp, n=np)
                    if m == mp and n == np_:
                        # Manually add the band edges
                        H_k_part[0, 0] += 0.0  # Conduction band edge
                        H_k_part[1, 1] += -0.3622 # Valence band edge
                    
                    # 4. Map to the big matrix
                    for a in range(nband):
                        for b in range(nband):
                            H[idx(a, m, n), idx(b, mp, np_)] = H_k_part[a, b]
    
    E, V = eigh(H)
    
    #----------------------------
    # Plotting the subbands
    #----------------------------
    plt.figure(figsize=(4, 6))
    for energy in E:
        plt.hlines(energy, 0, 1, linewidth=1, alpha=0.5, color='tab:blue')
    
    plt.xlim(0, 1)
    plt.xticks([])
    plt.ylim(-1, 1)
    plt.ylabel("Energy (eV)")
    plt.title(r"Subband energies")
    plt.axhline(0, color='red', linestyle='-', alpha=0.3)
    # plt.axhline(Eg, color='red', linestyle='-', alpha=0.3)
    plt.text(0.1, 0.9, f"L = {L}")
    plt.text(0.1, 0.8, f"N = {Ny}")
    plt.tight_layout()
    plt.savefig(f"subbands_vb_vb_toy_L{L}_Ny{Ny}.png", dpi = 300)
    plt.show()



#%%
#---------------------------------------------
# check for spurious states at a large k value
#---------------------------------------------
 
def random_unit_vector():
    """A random normalised unit vector"""
    v = np.random.randn(3)
    return v / np.linalg.norm(v)



N = 500 # checks the spurious states for N number of random momenta

bad_sign_count = 0
small_gap_count = 0

tol = 1e-6
k0 = 1e5   # very large momentum
for i in range(N):
    direction = random_unit_vector()
    kx, ky, kz = k0 * direction
    
    evals, _ = eigh(H_simple(kx, ky, kz))
    # print(evals)
    
    # Count positive/negative
    n_neg = np.sum(evals < -tol)
    
        
    # Check sign condition
    if not n_neg == 2:
        bad_sign_count += 1
        print(f"[FAIL SIGN] Iter {i}: n_neg={n_neg}")
    
    
    # Debug print if something fails
    if n_neg != 2:
        print("Eigenvalues:", evals)
        print("-"*50)

print("\n===== SUMMARY =====")
print(f"Sign failures     : {bad_sign_count} / {N}")
print(f"Small gap failures: {small_gap_count} / {N}")

