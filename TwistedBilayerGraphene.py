import numpy as np
from IPython.display import display
from sympy import Matrix
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# constants
omega_1 = 110 # 110meV 
a = 0.142 #0.142nm 
a_0 = np.sqrt(3)*a
k_D = 4*np.pi/3/a_0 # [1/nm]
phi = 2*np.pi/3
cis_phi = np.exp(1j*(phi))
hbar = 6.582e-13 # [meV s]
t =  2.7e3 # 2.7 eV -> meV 
v_F_hbar = 3/2 * a * t # hbar * v_F [meV nm] 
v_F = v_F_hbar/hbar # [nm/s]
# v_F = 1e15 # [nm/s]
# v_F_hbar = 1e15*hbar # hbar * v_F [meV nm] 


# Pauli matrices
sig_0 = np.array([[1,0],
                  [0,1]])
sig_x = np.array([[0,1],
                  [1,0]])
sig_y = np.array([[0, -1j], 
                  [1j, 0]], dtype=complex)

sig_z = np.array([[1,0],
                 [0,-1]])

class TBG():
    def __init__(self, theta):
        self.theta = theta # in radian
        self.k_theta = 2*k_D*np.sin(theta/2)
        self.alpha = omega_1/v_F_hbar/self.k_theta
        self.omega_1 = omega_1
        # self.omega_0 = omega_1 # BM model 
        self.omega_0 = 0 # CSCM model 

        # Hopping distance
        self.q = (np.nan,
                  self.k_theta*np.array([0, -1]), 
                  self.k_theta*np.array([np.sqrt(3)/2, 0.5]),
                  self.k_theta*np.array([-np.sqrt(3)/2, 0.5]))

    # Tunneling term
    @property
    def T(self):
        T = [np.nan] + [self.omega_0*sig_0 + self.omega_1*(np.cos(phi*j)*sig_x+np.sin(phi*j)*sig_y) for j in range(3)]
        # T = [np.nan] + [np.zeros((2,2))]*3
        return T


    # Monolayer term
    def h(self, k, theta):
        if k[0] == 0:
            theta_k = np.pi/2
        else: 
            theta_k = np.arctan2(k[1],k[0])
        e = np.exp(1j*(theta_k-theta))
        h = np.array([[0, e], [np.conj(e), 0]])
        return v_F_hbar*np.linalg.norm(k)*h 

    def eight_band_h(self, k):
        k = np.array(k)
        H = np.zeros((8,8), dtype=complex)
        H[0:2,0:2] = self.h(k, self.theta/2)
        
        # print(self.T)
        for i in range(2, 8, 2):
            
            H[i:i+2,i:i+2] = self.h(k+self.q[i//2], -self.theta/2)
            H[0:2, i:i+2] = self.T[i//2]
            H[i:i+2, 0:2] = self.T[i//2].conj().T
        return H
    
    def eight_band_h_reverse(self, k):
        k = np.array(k)
        H = np.zeros((8,8), dtype=complex)
        H[0:2,0:2] = self.h(k, -self.theta/2)
        
        # print(self.T)
        for i in range(2, 8, 2):
            
            H[i:i+2,i:i+2] = self.h(k-self.q[i//2], -self.theta/2)
            H[0:2, i:i+2] = self.T[i//2]
            H[i:i+2, 0:2] = self.T[i//2].conj().T
        return H

b_1 = np.array([0, -1])
b_2 = np.array([np.sqrt(3)/2, 1/2])
k_cutoff = 10

# Compute the distance from origin in k space with b as PLV
def k_distance(x, y):
    return np.linalg.norm((float(x)*b_1) + (float(y)*b_2))
k_distance = np.vectorize(k_distance)

def generate_pairs_numpy(a_range, b_range):
    a = np.arange(a_range[0], a_range[1] + 1)
    b = np.arange(b_range[0], b_range[1] + 1)
    A, B = np.meshgrid(a, b)
    return np.column_stack((A.ravel(), B.ravel()))

# Return the (kx, ky) vector from coord including offset
def coord_to_vec(k_coord, coord_arr):
    # k_vecx = (k_coord[0]+coord_arr[:,0])*b_1[0]+(k_coord[1] + coord_arr[:,1])*b_2[0]
    # k_vecy = (k_coord[0]+coord_arr[:,0])*b_1[1]+(k_coord[1] + coord_arr[:,1])*b_2[1]
    k_vecx = (coord_arr[:,0])*b_1[0]+(coord_arr[:,1])*b_2[0] + k_coord[0]
    k_vecy = (coord_arr[:,0])*b_1[1]+(coord_arr[:,1])*b_2[1] + k_coord[1]
    return k_vecx, k_vecy 


# Cutoff with both layers
class TBG2():
    def __init__(self, theta, k_cutoff=k_cutoff):
        self.theta = theta # in radian
        self.k_theta = 2*k_D*np.sin(theta/2)
        self.alpha = omega_1/v_F_hbar/self.k_theta
        self.omega_1 = omega_1
        self.omega_0 = omega_1 # BM model 
        # self.omega_0 = 0 # CSCM model 
        self.k_cutoff = k_cutoff
        self.k_set = self.gen_k_set()
        
        self.H_T = self.Tunneling()

    def gen_k_set(self):
        k_coord = (0,0)
        coord_list = generate_pairs_numpy((-2*int(self.k_cutoff), 2*int(self.k_cutoff)), (-2*int(self.k_cutoff), 2*int(self.k_cutoff)))
        k_set = coord_list[k_distance(coord_list[:,0], coord_list[:,1])<self.k_cutoff]
        return k_set
    
    def gen_lookup_arr(self):
        k_set_list = self.k_set.tolist()
        lookup_arr = np.full((len(self.k_set), 7), -1, dtype=int)
        for i in range(lookup_arr.shape[0]):
            i_1 = i + 1 if k_distance(self.k_set[i,0]+1, self.k_set[i,1]) < self.k_cutoff else -1
            i_2 = k_set_list.index(list(self.k_set[i,:] + np.array([0,1]))) if k_distance(self.k_set[i,0], self.k_set[i,1]+1) < self.k_cutoff else -1
            i_3 = k_set_list.index(list(self.k_set[i,:] + np.array([-1,-1]))) if k_distance(self.k_set[i,0]-1, self.k_set[i,1]-1) < self.k_cutoff else -1
            i_4 = i - 1 if k_distance(self.k_set[i,0]-1, self.k_set[i,1]) < self.k_cutoff else -1
            i_5 = k_set_list.index(list(self.k_set[i,:]- np.array([0,1]))) if k_distance(self.k_set[i,0], self.k_set[i,1]-1) < self.k_cutoff else -1
            i_6 = k_set_list.index(list(self.k_set[i,:] - np.array([-1,-1]))) if k_distance(self.k_set[i,0]+1, self.k_set[i,1]+1) < self.k_cutoff else -1
            lookup_arr[i,:] = np.array([i, i_1, i_2, i_3, i_4, i_5, i_6])
        
        # test_index = 189
        # k_coord = (0,0) 
        # plt.figure(figsize=(6,6))
        # plt.plot(self.k_set[:,0]*b_1[0]+self.k_set[:,1]*b_2[0], self.k_set[:,0]*b_1[1]+self.k_set[:,1]*b_2[1], 'o', ms=1)
        # for j in range(7):
        #     if lookup_arr[test_index,j] == -1: continue 
        #     plt.plot(self.k_set[lookup_arr[test_index,j],0]*b_1[0]+self.k_set[lookup_arr[test_index,j],1]*b_2[0], self.k_set[lookup_arr[test_index,j],0]*b_1[1]+self.k_set[lookup_arr[test_index,j],1]*b_2[1], 'o', ms=2, label=j)
        # plt.title(f"k cut_off = {k_cutoff}, k_0 = {k_coord}, self.k_set size = {len(self.k_set)}")
        # plt.legend()
        # plt.show()

        return lookup_arr

    def Tunneling(self):
        zero2 = np.zeros((2,2))
        lookup_arr = self.gen_lookup_arr()
        e = cis_phi
        ec = np.conj(e)
        # T = [np.nan] + [np.block([[zero2, self.omega_0*sig_0 + self.omega_1*(np.cos(phi*j)*sig_x+np.sin(phi*j)*sig_y)],[zero2, zero2]]) for j in range(3)] # Tunneling amplitudes
        T = [np.nan, 
             np.block([[zero2, np.ones((2,2))], [zero2, zero2]])*self.omega_1, 
             np.block([[zero2, np.array([[ec, 1], [e, ec]])], [zero2, zero2]])*self.omega_1, 
             np.block([[zero2, np.array([[e, 1], [ec, e]])], [zero2, zero2]])*self.omega_1] # Tunneling amplitudes
        # for i in range(1,4):
        #     display(Matrix(T[i]))
        H = np.zeros((4*len(self.k_set), 4*len(self.k_set)), dtype=complex)
        # Loop over 4x4 block entries
        for i in range(len(self.k_set)):
            for j in range(1,4):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[4*i:4*(i+1), 4*l:4*(l+1)] = T[j]
            for j in range(4,7):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[4*i:4*(i+1), 4*l:4*(l+1)] = T[j-3].conj().T
        return H

    def h_k(self, k_vecx, k_vecy, theta):
        theta_k = np.arctan2(k_vecy,k_vecx)
        e1 = np.exp(1j*(theta_k-theta))
        h1 = np.array([[0, e1], [np.conj(e1), 0]])
        e2 = np.exp(1j*(theta_k+theta))
        h2 = np.array([[0, e2], [np.conj(e2), 0]])
        # print(v_F_hbar*self.k_theta)
        return v_F_hbar*self.k_theta*np.linalg.norm([k_vecx, k_vecy])*block_diag(h1,h2) 

    def Hamiltonian(self, k_coord):
        k_vecx, k_vecy = coord_to_vec(k_coord, self.k_set)
        # print(np.column_stack((k_vecx, k_vecy)))
        diagonal_list = [self.h_k(kx, ky, self.theta/2) for kx,ky in zip(k_vecx, k_vecy)]
        H_T = self.Tunneling()
        H_diag = block_diag(*diagonal_list) 
        H = H_T + H_diag
        return H

def label_layer(coord_arr):
    nn_vec_top = [np.array([1,0]), np.array([0,1]), np.array([-1,-1])]
    nn_vec_bottom = [np.array([-1,0]), np.array([0,-1]), np.array([1,1])]
    nn_vec = [nn_vec_top, nn_vec_bottom]
    coord_list = coord_arr.tolist()
    # print(coord_list)
    layer_label = np.full(coord_arr.shape[0], -1)
    # print(coord_list)
    i = coord_list.index([0,0])
    layer_label[i] = 0 # 0 for Top layer
    # locate unvisited neighbors of (0,0)
    N_0 = []
    for vec in nn_vec[layer_label[i]]:
        if (coord_arr[i] + vec).tolist() in coord_list:
            N_0.append((coord_arr[i] + vec).tolist())
    flag = 0
    while N_0 != []:
        # print(N_0)
        N_0_new = []
        for n in N_0:
            i = coord_list.index(n)
            # print(n, layer_label[i])   
            if layer_label[i] != -1:
                # n has been visited
                continue
            layer_label[i] = 1-flag
            # print(n, layer_label[i])
            # locate unvisited neighbors of n
            for vec in nn_vec[layer_label[i]]:
                if (coord_arr[i] + vec).tolist() in coord_list:
                    N_0_new.append((coord_arr[i] + vec).tolist())
        flag = 1-flag
        N_0 = N_0_new
    return layer_label

# Cutoff with top layer:
class TBG3():
    def __init__(self, theta, k_cutoff=k_cutoff):
        self.theta = theta # in radian
        self.k_theta = 2*k_D*np.sin(theta/2)
        self.alpha = omega_1/v_F_hbar/self.k_theta
        self.omega_1 = omega_1
        # self.omega_0 = omega_1 # BM model 
        # self.omega_0 = 0 # CSCM model 
        self.k_cutoff = k_cutoff
        self.k_set, self.layer_label = self.gen_k_set()
        # print(self.k_set, self.layer_label)
        self.H_T = self.Tunneling()

        

    def gen_k_set(self):
        # k_coord = (0,0)
        coord_list = generate_pairs_numpy((-2*int(self.k_cutoff), 2*int(self.k_cutoff)), (-2*int(self.k_cutoff), 2*int(self.k_cutoff)))
        k_set = coord_list[k_distance(coord_list[:,0], coord_list[:,1])<self.k_cutoff]
        layer_label = label_layer(k_set)
        # Filter k_set to remove unwanted points
        k_set = k_set[np.where(layer_label != -1)]
        # k_vecx, k_vecy = coord_to_vec(k_coord, k_set)
        layer_label = layer_label[np.where(layer_label != -1)]
        return k_set, layer_label
    
    def gen_lookup_arr(self):
        k_set_list = self.k_set.tolist()
        lookup_arr = np.full((len(self.k_set), 7), -1, dtype=int)
        for i in range(lookup_arr.shape[0]):
            # print(self.k_set[i,:], self.layer_label[i])
            if self.layer_label[i] == 0:
                i_1 = i + 1 if k_distance(self.k_set[i,0]+1, self.k_set[i,1]) < self.k_cutoff else -1
                i_2 = k_set_list.index(list(self.k_set[i,:] + np.array([0,1]))) if k_distance(self.k_set[i,0], self.k_set[i,1]+1) < self.k_cutoff else -1
                i_3 = k_set_list.index(list(self.k_set[i,:] + np.array([-1,-1]))) if k_distance(self.k_set[i,0]-1, self.k_set[i,1]-1) < self.k_cutoff else -1
                i_4, i_5, i_6 = -1, -1, -1
            else:
                i_4 = i - 1 if k_distance(self.k_set[i,0]-1, self.k_set[i,1]) < self.k_cutoff else -1
                i_5 = k_set_list.index(list(self.k_set[i,:]- np.array([0,1]))) if k_distance(self.k_set[i,0], self.k_set[i,1]-1) < self.k_cutoff else -1
                i_6 = k_set_list.index(list(self.k_set[i,:] - np.array([-1,-1]))) if k_distance(self.k_set[i,0]+1, self.k_set[i,1]+1) < self.k_cutoff else -1
                i_1, i_2, i_3 = -1, -1, -1
            lookup_arr[i,:] = np.array([i, i_1, i_2, i_3, i_4, i_5, i_6])
        return lookup_arr

    def Tunneling(self):
        lookup_arr = self.gen_lookup_arr()
        e = cis_phi
        ec = np.conj(e)
        T = [np.nan] + [self.omega_0*sig_0 + self.omega_1*(-np.cos(phi*j)*sig_x+np.sin(phi*j)*sig_y) for j in range(3)] # Tunneling amplitudes
        # T = [np.nan, 
        #      np.ones((2,2))*self.omega_1, 
        #      np.array([[ec, 1], [e, ec]])*self.omega_1, 
        #      np.array([[e, 1], [ec, e]])*self.omega_1] # Tunneling amplitudes
        H = np.zeros((2*len(self.k_set), 2*len(self.k_set)), dtype=complex)
        # Loop over 4x4 block entries
        for i in range(len(self.k_set)):
            for j in range(1,4):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[2*i:2*(i+1), 2*l:2*(l+1)] = T[j]
            for j in range(4,7):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[2*i:2*(i+1), 2*l:2*(l+1)] = T[j-3].conj().T

        return H
    
    def h_k(self, k_vecx, k_vecy, i, theta):
        theta_k = np.arctan2(k_vecy,k_vecx)
        e1 = np.exp(1j*(theta_k-theta))
        h1 = np.array([[0, e1], [np.conj(e1), 0]])
        e2 = np.exp(1j*(theta_k+theta))
        h2 = np.array([[0, e2], [np.conj(e2), 0]])
        h = h1 if i == 0 else h2
        return v_F_hbar*self.k_theta*np.linalg.norm([k_vecx, k_vecy])*h
    
    def Hamiltonian(self, k_coord):
        k_vecx, k_vecy = coord_to_vec(k_coord, self.k_set)
        diagonal_list = [self.h_k(kx, ky, i, self.theta/2) for kx,ky,i in zip(k_vecx, k_vecy, self.layer_label)]
        H_diag = block_diag(*diagonal_list) 
        H = self.H_T + H_diag
        return H
    
class BM_original(TBG3):
    def __init__(self, theta, k_cutoff=k_cutoff):
        self.omega_0 = omega_1 # BM model

        super().__init__(theta, k_cutoff)
    
    def Tunneling(self):
        lookup_arr = self.gen_lookup_arr()
        e = cis_phi
        ec = np.conj(e)
        # T = [np.nan] + [self.omega_0*sig_0 + self.omega_1*(np.cos(phi*j)*sig_x+np.sin(phi*j)*sig_y) for j in range(3)] # Tunneling amplitudes
        T = [np.nan, 
             np.ones((2,2))*self.omega_1, 
             np.array([[ec, 1], [e, ec]])*self.omega_1, 
             np.array([[e, 1], [ec, e]])*self.omega_1] # Tunneling amplitudes
        H = np.zeros((2*len(self.k_set), 2*len(self.k_set)), dtype=complex)
        # Loop over 4x4 block entries
        for i in range(len(self.k_set)):
            for j in range(1,4):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[2*i:2*(i+1), 2*l:2*(l+1)] = T[j]
            for j in range(4,7):
                l = lookup_arr[i,j]
                if l == -1: continue
                H[2*i:2*(i+1), 2*l:2*(l+1)] = T[j-3].conj().T

        return H
    
class BMCM(TBG3):
    def __init__(self, theta, k_cutoff=k_cutoff):
        self.omega_0 = omega_1
        super().__init__(theta, k_cutoff)

class CSCM(TBG3):
    def __init__(self, theta, k_cutoff=k_cutoff, sublattice_potential=0):
        self.omega_0 = 0
        self.sublattice_potential = sublattice_potential
        super().__init__(theta, k_cutoff)
    
    def h_k(self, k_vecx, k_vecy, i, theta):
        theta_k = np.arctan2(k_vecy,k_vecx)
        e1 = np.exp(1j*(theta_k-theta))
        h1 = np.array([[0, e1], [np.conj(e1), 0]])
        e2 = np.exp(1j*(theta_k+theta))
        h2 = np.array([[0, e2], [np.conj(e2), 0]])
        h = h1 if i == 0 else h2
        return v_F_hbar*self.k_theta*np.linalg.norm([k_vecx, k_vecy])*h + self.sublattice_potential*sig_z


# k-path
A, B, C, D = (0,1), (0,0), (0,-1), (np.sqrt(3)/2, 0.5)
# A, B, C, D = (0,2), (0,1), (0,0), (np.sqrt(3)/2, 1.5)
kpath_x = np.concatenate((np.linspace(A[0], B[0], 10), np.linspace(B[0], C[0], 10), np.linspace(C[0], D[0], 20), np.linspace(D[0], A[0], 10)))
kpath_y = np.concatenate((np.linspace(A[1], B[1], 10), np.linspace(B[1], C[1], 10), np.linspace(C[1], D[1], 20), np.linspace(D[1], A[1], 10)))
kpath = (kpath_x, kpath_y)

if __name__ == '__main__':
    # print(omega_1*1e-6*1.6e-19/hbar/v_F)
    tbg = TBG2(0.03)
    print(tbg.k_theta)
    matrix = tbg.Hamiltonian((0,0))
    print(matrix[0,1])
