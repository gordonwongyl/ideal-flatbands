import numpy as np
import matplotlib.pyplot as plt


# return a q x q matrix H(k) corresponding to Harper equation
def Hamiltonian_k(k1, k2, a, p, q=3, t=1):
    M = np.zeros((q,q), dtype=np.complex128)
    for i in range(q):

        M[i,i] = -2*t*np.cos(k1*a + 2*np.pi*p/q * (i+1))
        M[(i+1)%q,i] = -t*np.exp(1j*k2*a)  
        M[i,(i+1)%q] = -t*np.exp(-1j*k2*a)

    return M

def diagonlaize_pool(inputs):
        p, q, N1, N2, a, t, filename = inputs
        print(p, q, N1, N2, a, t, filename)
        diagonalize(p, q, N1, N2, a, t, file=filename)

def diagonalize(p, q, N1, N2, a, t, file=None):
    k1_list = np.arange(-1/q, 1/q+1/N1, 2/N1)*np.pi/a
    # k2_list = np.arange(-1, 1, 2/N2)*np.pi/a # whole magnetic BZ
    k2_list = np.arange(-1/q, 1/q+1/N2, 2/N2)*np.pi/a # only ky = [-pi/qa, pi/qa]
    eigenvalues_list = []
    eigenvector_list = []
    # Hamiltonian_k_list = []
    
    for k1 in k1_list:
        eigvecs_horizontal = []
        eigvals_horizontal = []
        # Hamiltonian_horizontal = []
        for k2 in k2_list:
            # k_pair_list.append((k1,k2))
            M = Hamiltonian_k(k1, k2, a, p, q, t)

            eigval, eigvec = np.linalg.eig(M)

            # sort the results into q bands
            idx = eigval.argsort()
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]

            # Hamiltonian_horizontal.append(M)
            eigvals_horizontal.append(eigval)
            eigvecs_horizontal.append(eigvec)

        # Hamiltonian_k_list.append(Hamiltonian_horizontal)
        eigenvalues_list.append(eigvals_horizontal)
        eigenvector_list.append(eigvecs_horizontal)

    # Hamiltonian_k_arr = np.array(Hamiltonian_k_list)
    eigenvalues_arr = np.array(eigenvalues_list)
    eigenvector_arr = np.array(eigenvector_list)
    # Zone = construct_BZ(eigenvector_arr, a, p, q)
    # print(eigenvector_arr.shape)
    # np.save(file+f"/{N1}_by_{N2}_p_{p}_q_{q}_H", Hamiltonian_k_arr) # shape: Hk = [k1, k2, :, :]
    np.save(file+f"/{N1}_by_{N2}_p_{p}_q_{q}_eigenvalues", eigenvalues_arr) # shape: Enk = [k1 = 1 ... N1/q , k2 = 1... N2, n = 1...q]
    np.save(file+f"/{N1}_by_{N2}_p_{p}_q_{q}_eigenvectors", eigenvector_arr) # shape: a_n = [k1 = 1 ... N1/q , k2 = 1... N2, :, n = 1...q]
    # np.save(file+f"/{N1}_by_{N2}_p_{p}_q_{q}_BZ", Zone) # shape: unk = [n = 1...q, k1 = 1 ... N1/q , k2 = 1... N2, :]

def index_max_norm(vec:np.ndarray):
    norm = np.sqrt((vec.conj()*vec).real)
    index = np.argsort(norm)
    return index[-1]

# Fix the phase such that the ith entry is real
def fix_phase(vec:np.ndarray, i):
    entry = vec[i]
    vec = (entry.conj()*entry)**0.5 * vec/entry
    return vec

# Compute the tensor with the gauge_invariant formula; take BZ as the bloch band of eigenvectors a_n[:,:,:,n]
# Hopefully, imaginary part is the same as the curvature 
def geometric_tensor(BZ: np.ndarray, dk1=None, dk2=None):
    n1, n2, _ = BZ.shape
    Q = np.zeros((2,2, n1, n2), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            k = BZ[i,j,:]
            k_dk = (BZ[(i+1)%n1, j, :], BZ[i, (j+1)%n2, :])

            for u in range(2):
                for v in range(2):
                    A = np.inner(k_dk[u].conj(), k_dk[v])
                    B = np.inner(k_dk[v].conj(), k)
                    C = np.inner(k.conj(), k_dk[u])
                    D = np.inner(k.conj(), k_dk[v])
                    E = np.inner(k_dk[v].conj(), k)

                    Q[u,v, i, j] = np.log(A*B*C*D*E)
    return Q

def curvature(BZ: np.ndarray, dk1=None, dk2=None):
    n1, n2, _ = BZ.shape
    F = np.zeros((n1, n2), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            k = BZ[i,j,:]
            k_dk = (BZ[(i+1)%n1, j, :], BZ[i, (j+1)%n2, :], BZ[(i+1)%n1, (j+1)%n2, :])
            
            A = np.inner(k.conj(), k_dk[0])
            B = np.inner(k_dk[0].conj(), k_dk[2])
            C = np.inner(k_dk[2].conj(), k_dk[1])
            D = np.inner(k_dk[1].conj(), k)
            F[i, j] = np.log(A*B*C*D)
    return F

         

if __name__ == "__main__":
    N1, N2 = 18,18
    q = 3
    p = 1
    a = 0.5
    t = 1
    k1_list = np.arange(-1/q, 1/q, 2/N1)*np.pi/a
    k2_list = np.arange(-1, 1, 2/N2)*np.pi/a
    dict = {}

    eigenvalues_list = []
    eigenvector_list = []
    Hamiltonian_k_list = []
    k_pair_list = []

    for k1 in k1_list:
        eigvecs_horizontal = []
        eigvals_horizontal = []
        Hamiltonian_horizontal = []
        for k2 in k2_list:
            # k_pair_list.append((k1,k2))
            M = Hamiltonian_k(k1, k2, a, p, q, t)

            eigval, eigvec = np.linalg.eig(M)

            # sort the results into q bands
            idx = eigval.argsort()
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]

            Hamiltonian_horizontal.append(M)
            eigvals_horizontal.append(eigval)
            eigvecs_horizontal.append(eigvec)

        Hamiltonian_k_list.append(Hamiltonian_horizontal)
        eigenvalues_list.append(eigvals_horizontal)
        eigenvector_list.append(eigvecs_horizontal)

    Hamiltonian_k_arr = np.array(Hamiltonian_k_list)
    eigenvalues_arr = np.array(eigenvalues_list)
    eigenvector_arr = np.array(eigenvector_list)
    # Zone = construct_BZ(eigenvector_arr, a, p, q)
    # print(eigenvector_arr.shape)
    np.save(f"{N1}_by_{N2}_p_{p}_q_{q}_H", Hamiltonian_k_arr) # shape: Hk = [k1, k2, :, :]
    np.save(f"{N1}_by_{N2}_p_{p}_q_{q}_eigenvalues", eigenvalues_arr) # shape: Enk = [k1 = 1 ... N1/q , k2 = 1... N2, n = 1...q]
    np.save(f"{N1}_by_{N2}_p_{p}_q_{q}_eigenvectors", eigenvector_arr) # shape: a_n = [k1 = 1 ... N1/q , k2 = 1... N2, :, n = 1...q]
    # np.save(f"{N1}_by_{N2}_p_{p}_q_{q}_BZ", Zone) # shape: unk = [n = 1...q, k1 = 1 ... N1/q , k2 = 1... N2, :]
    print("Finished")



# compute the partial derivative on the Brullion Zone for band n; unk = BZ[k1, k2, :]
def partial_derivative_BZ(BZ: np.ndarray, dk1, dk2):
    n1, n2, N = BZ.shape
    dx_BZ = np.zeros((n1,n2,N), dtype=complex)
    dy_BZ = np.zeros((n1,n2, N), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            idx = index_max_norm(BZ[i, j,:])

            # dx_BZ[i,j,:] = (fix_phase(BZ[(i+1)%n1, j,:], idx) - fix_phase(BZ[(i-1)%n1, j,:], idx))/2/dk1 # Three point formula for differentiation
            # dy_BZ[i,j,:] = (fix_phase(BZ[i, (j+1)%n2,:], idx) - fix_phase(BZ[i, (j-1)%n2,:], idx))/2/dk2 # Three point formula for differentiation
            dx_BZ[i,j,:] = (fix_phase(BZ[(i+1)%n1, j,:], idx) - fix_phase(BZ[i, j,:], idx))/dk1 # Two point formula for differentiation
            dy_BZ[i,j,:] = (fix_phase(BZ[i, (j+1)%n2,:], idx) - fix_phase(BZ[i, j,:], idx))/dk2 # Two point formula for differentiation
    return dx_BZ, dy_BZ

# compute the berry connection; Ai(k) = A[i, k1, k2]
def berry_connection(BZ: np.ndarray, dk1, dk2, dx_BZ=None, dy_BZ=None):
    n1, n2, _ = BZ.shape
    A = np.zeros((2, n1, n2), dtype=complex) # A_x = A[0,:,:], A_y = A[1,:,:]
    if dx_BZ is None or dy_BZ is None:
        dx_BZ, dy_BZ = partial_derivative_BZ(BZ, dk1, dk2)

    for i in range(n1):
        for j in range(n2):
            A[0,i,j] = -1j * np.inner(BZ[i, j, :].conj(), dx_BZ[i,j,:])
            A[1,i,j] = -1j * np.inner(BZ[i, j, :].conj(), dy_BZ[i,j,:])
    return A

# compute the 2x2 geometric tensor; Qij (k) = Q[i,j, k1, k2]
def geometric_tensor2(BZ: np.ndarray, dk1, dk2):
    n1, n2, _ = BZ.shape
    Q = np.zeros((2,2, n1, n2), dtype=complex)
    dx_BZ, dy_BZ = partial_derivative_BZ(BZ, dk1, dk2)
    A = berry_connection(BZ, dk1, dk2, dx_BZ=dx_BZ, dy_BZ=dy_BZ)
    for i in range(n1):
        for j in range(n2):
            Q[0,0, i, j] =  np.inner(dx_BZ[i,j,:].conj(), dx_BZ[i,j,:]) - A[0,i,j]*A[0,i,j]
            Q[0,1, i, j] =  np.inner(dx_BZ[i,j,:].conj(), dy_BZ[i,j,:]) - A[0,i,j]*A[1,i,j]
            Q[1,0, i, j] =  np.inner(dy_BZ[i,j,:].conj(), dx_BZ[i,j,:]) - A[1,i,j]*A[0,i,j]
            Q[1,1, i, j] =  np.inner(dy_BZ[i,j,:].conj(), dy_BZ[i,j,:]) - A[1,i,j]*A[1,i,j]

    return Q

# compute FSM with the norm formula (eq. 5 in report)
def FSM(BZ: np.ndarray):
    n1, n2, _ = BZ.shape
    g = np.zeros((2,2, n1, n2))
    for i in range(n1):
        for j in range(n2):
            k = BZ[i,j,:]
            k_dk = (BZ[(i+1)%n1, j, :], BZ[i, (j+1)%n2, :], BZ[(i+1)%n1, (j+1)%n2, :])
            A = np.abs(np.inner(k.conj(), k_dk[0]))**2
            B = np.abs(np.inner(k.conj(), k_dk[1]))**2
            C = np.abs(np.inner(k.conj(), k_dk[2]))**2
            g[0, 0, i, j] = 1 - A
            g[0, 1, i, j] = 0.5*(A+B-C-1)
            g[1, 0, i, j] = 0.5*(A+B-C-1) 
            g[1, 1, i, j] = 1 - B
    return g