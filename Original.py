# This is a program to calculate the Faraday Polarization Rotation. We will focuse o the Verdet Constant of Rb vapor.
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sympy.physics.wigner import wigner_3j


def atomic_constant(Atom, p, n, start):
    '''
    Atom: Character string designating the atomic symbol
    P: Real Linear array to store atomic constants upon return
    N: Integer equal to the index of the isotope to be used
    start: Integer equal to 1 if the first time the function is called
    '''
    # Create a 12x8 zero matrix
    C = np.zeros((12, 8))

    # Fill columns with atomic data for 'LI006', 'LI007', 'NA023', 'K0039', 'K0041', 'RB085', 'RB087', 'CS133'
    # [upper level energy(cm-1), (cm-1), ap(cm-1), ac(cm-1), bp(cm-1), lower level energy(cm-1) ]
    C[0:12, 0] = [14903.89, 0.223559, 1.6918e-4, -1.2882e-4, -3.6860e-7, 0.0, 0.0050747387200, 0.8220300, 1.0, 26.4, 26.4,
         0.0742]
    C[0:12, 1] = [14903.89, 0.22355900, 4.4676e-4, -3.4017e-4, -1.8430e-5, 0.0, 0.0134010049, 3.2563600, 1.5, 26.4, 26.4,
         0.9258]
    C[0:12, 2] = [16967.647, 11.464, 1.177e-3, -5.837e-6, 2.35e-4, 0.0, 0.0295475433, 2.2174000, 1.5, 16.0, 16.0, 1.0]
    C[0:12, 3] = [13023.65, 38.480, 3.642e-4, 8.8950e-6, 2.310e-4, 0.0, 0.00770065604, 0.3914300, 1.5, 26.0, 26.0, 0.9312]
    C[0:12, 4] = [13023.6500, 38.48000, 2.004e-4, 4.894e-6, 2.820e-4, 0.0, 0.00423649534, 0.21487, 1.5, 26.0, 26.0, 0.0688]
    C[0:12, 5] = [12737.36, 158.4000, 1.525e-3, 2.496e-5, 2.0864e-3, 0.0, 0.0337537115, 1.3524, 2.5, 25.5, 28.1, 0.7215]
    C[0:12, 6] = [12737.36, 158.40, 5.149e-3, 8.428e-5, 1.0432e-3, 0.0, 0.113990236, 2.7500, 1.5, 25.5, 28.1, 0.2785]
    C[0:12, 7] = [11547.65, 369.41, 3.5692e-3, -2.254e-4, -3.2e-5, 0.0, 0.0766582975, 2.5779, 3.5, 29.9, 34.0, 1.0]

    FT = 1.499e9
    Label = ['LI006', 'LI007', 'NA023', 'K039', 'K041', 'RB085', 'RB087', 'CS133']

    # Find the correct isotope based on Atom string
    isotope_found = False
    for i in range(8):
        if Atom == Label[i][:2]:
            isotope_found = True
            if n == 0:
                k = i  # Select first isotope of the atom
            else:
                k = min(i + n, 7)  # For subsequent isotopes
            break

    if not isotope_found:
        print(f"Atom {Atom} not found in the Label list.")
        return p

    # Fill the array `p` with constants from matrix `C`
    for i in range(11):
        p[i] = C[i, k]

    # Perform computations on `p` array
    p[12] = np.sqrt(FT / (p[0] - p[1]) ** 3 / p[10])
    p[13] = np.sqrt(2.0 * FT / (p[0] + 0.5 * p[1]) ** 3 / p[9])
    p[14] = FT / (p[0] - p[1]) ** 2 / p[10]
    p[15] = 2.0 * FT / (p[0] + 0.5 * p[1]) ** 2 / p[9]
    p[16] = C[11, k]

    print(f"ATCTS was called for {Atom} with isotope index {n}.")
    return p

def THJ(j_1, j_2, j_3, m_1, m_2, m_3):
    return float(wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3))


def level(p, nd, scratch):
    #
    # p is the atomic constants,
    # #nd is the total number of transition frequency and f-value pairs

    # Initialization
    z = np.zeros(5) # Initialize Z array
    d = np.zeros(8)  # Initialize d array
    u = np.zeros((8, 8))  # Initialize u matrix,
    w = np.zeros((8, 9))  # Initialize w matrix
    h = np.zeros((8, 8))  # Initialize h matrix, hamiltonian matrix
    index = np.zeros(8, dtype=int)  # Initialize index

    global nlb, lb

    nh=8 # how many levels (except zeeman splitting) in total including upper and lower level 2S1/3 2P1/2, 2P3/2

    z[0] = 1.0 / np.sqrt(2.0 * p[8] + 1.0)
    s = p[8] + 1.5 # biggest possible valve for MJ+MI
    m = int( 2 * s + 1)

    # label of the all states including the magnetic sub-levels.
    label(p[8])

    # Sub-block diagonalization of Hamiltonian matrix:
    for i in range(1, m + 1):  # range from 1 to m included, which means explore through all m MJ+MI blocks: as Na23 I=3/2 as example, it means MJ+MI from 3,2,1, 0, -1, -2, -3
        nl = 0  # Initialize it as 0, lower index in lb for MJ+MI=s
        nu = 0  # Initialize it as 0
        n_f = 0  # index for transitions

        for j in range(1, nlb + 1): # search inside all states including magnetic sub-levels
            string = list(map(float, lb[j-1][0:21].split()))
            sp=string[4] # sp: get the valve of MJ+MI
            if sp > s:
                nl = j
            elif sp == s:
                nu = j
            elif sp < s:
                break


        nl += 1  #
        n = nu - nl + 1
        # nl is the lower position, and nu is the upper position

        # Diagonalize the MJ + MI block
        nh,h=Matrix(nh, h, p, nl, nu)
        print(f'H: {h}')

        d, u = jacobi_rotation(h)
        print(f'eigenvalues: {d}')
        print(f'eigenvectors: {u}')
        print(f'H after diagonlization: {h}')
        if i != 1:
            # Transition matrix in the initial basis
            for j in range(nls, nus + 1):
                ir = j - nls+1
                qi = list(map(float, lb[j - 1][0:16].split()))  # Assuming lb is a list of strings
                for k in range(nl, nu + 1):
                    ic = k - nl +1
                    qj = list(map(float, lb[k - 1][0:16].split()))
                    h[ir-1][ic-1] = 0.0

                    if not (qi[3] != qj[3] or qi[0] == qj[0]):
                        z[1] = THJ(float(qi[1]), 1.0, float(qj[1]), float(-qi[2]), float(qi[2] - qj[2]), float(qj[2]))
                        h[ir - 1][ic - 1] = z[0] * z[1]
                        l = qi[1] + qj[1] - qi[2] - 0.5

                        if l % 2 != 0:
                            h[ir - 1][ic - 1] = -h[ir - 1][ic - 1]

                        sp = max(qi[1], qj[1])

                        if sp == 0.5:
                            h[ir - 1][ic - 1] *= p[12]
                        elif sp == 1.5:
                            h[ir - 1][ic - 1] *= p[13]

            # Transition frequencies and f-values
            print (f'H before T and f calculation: {h}')

            for j in range(1, ns + 1):
                for k in range(1, n + 1):
                    z[1] = 0.0
                    for jj in range(1, ns + 1):
                        for kk in range(1, n + 1):
                            z[1] += w[jj - 1][j - 1] * h[jj - 1][kk - 1] * u[kk - 1][k - 1]
                    z[2] = w[j - 1][8] - d[k - 1]
                    z[3] = np.sign(z[2])
                    z[2] = abs(z[2])
                    z[1] = z[1] * z[1] * z[2]

                    if not (z[1] < 1e-12):
                        nd += 1

                        if z[3] > 0.0:
                            z[4] = 0.0
                            for l in range(1, n + 1):
                                string = list(map(float, lb[l + nl - 1][0:21].split()))
                                sp = string[2]  # sp: get the valve of MJ
                                z[4] += np.sign(sp) * u[l - 1][k - 1] ** 2
                        else:
                            z[4] = 0.0
                            for l in range(1, ns + 1):
                                sp = list(map(float, lb[l + nls - 1][0:21].split()))
                                sp = string[2]  # sp: get the valve of MJ
                                z[4] += np.sign(sp) * w[l - 1][j - 1] ** 2

                        scratch[n_f, :] = [z[2], z[1], z[3], np.sign(z[4])]
                        n_f = n_f + 1

        # Save eigenvalues and eigenvectors for subsequent calculation
        if i != m:
            ns = n
            nls = nl
            nus = nu

            for j in range(1, n + 1):
                w[j - 1][8] = d[j - 1]
                for k in range(1, n + 1):
                    w[k - 1][j - 1] = u[k - 1][j - 1]
            s = s - 1
    return p, nd, scratch


def max_offdiag_symmetric(A):
    """Find the largest off-diagonal element in the matrix A."""
    n = len(A)
    max_val = 0.0
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > abs(max_val):
                max_val = A[i, j]
                p, q = i, j
    return max_val, p, q


def jacobi_rotation(A, tol=1.0e-16):
    """Diagonalize a symmetric matrix using the Jacobi method."""
    n = len(A)
    D = np.copy(A)  # Make a copy of the original matrix
    V = np.eye(n)  # Identity matrix for eigenvectors

    for _ in range(100):  # Maximum number of iterations
        max_val, p, q = max_offdiag_symmetric(D)

        if abs(max_val) < tol:
            break  # Stop if off-diagonal elements are sufficiently small

        # Calculate the rotation angle
        if D[p, p] != D[q, q]:
            tau = (D[q, q] - D[p, p]) / (2 * D[p, q])
            t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau ** 2))
        else:
            t = 1

        c = 1 / np.sqrt(1 + t ** 2)
        s = t * c

        # Perform the rotation
        for i in range(n):
            if i != p and i != q:
                D_ip = D[i, p]
                D_iq = D[i, q]
                D[i, p] = D_ip * c - D_iq * s
                D[p, i] = D[i, p]
                D[i, q] = D_iq * c + D_ip * s
                D[q, i] = D[i, q]

        D_pp = D[p, p]
        D_qq = D[q, q]
        D_pq = D[p, q]

        D[p, p] = c ** 2 * D_pp - 2 * s * c * D_pq + s ** 2 * D_qq
        D[q, q] = s ** 2 * D_pp + 2 * s * c * D_pq + c ** 2 * D_qq
        D[p, q] = 0
        D[q, p] = 0

        # Update eigenvector matrix V
        for i in range(n):
            V_ip = V[i, p]
            V_iq = V[i, q]
            V[i, p] = V_ip * c - V_iq * s
            V[i, q] = V_iq * c + V_ip * s

    return np.diag(D), V  # Return eigenvalues and eigenvectors

def Matrix(NH, H, P, NL, NU):
    # Initialize constants and arrays
    BN = 2.5426228e-05  # cm-1/kgauss

    global nlb, lb

    # Define array R (3x3x3)
    R = np.array([
        [1.1449056155209570e-01, 0.0, 0.0],
        [0.0, 3.8075110222036970e-02, -5.4033883822915113e-02],
        [0.0, -5.4033883822915113e-02, -2.4122750777996062e-01],
        [1.2247448713915891e+00, 0.0, 0.0],
        [0.0, 3.2659863237109041e+00, 5.7735026918962577e-01],
        [0.0, 5.7735026918962577e-01, -2.0655911179772890e+00],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 1.0]]).reshape(3, 3, 3)

    W = np.zeros(5)
    QI = np.zeros(5)
    QJ = np.zeros(4)

    # Initializing W(4) and W(5)
    W[3] = math.sqrt(P[8] * (P[8] + 1.0) * (2.0 * P[8] + 1.0))
    W[4] = math.sqrt(0.2 * (P[8] + 1.0) * (2.0 * P[8] + 1.0) * (2.0 * P[8] + 3.0) / P[8] / (2.0 * P[8] - 1.0))
    QI[4] = P[8]

    # Main calculation of the Hamiltonian matrix elements for block MJ + MI
    for I in range(NL, NU + 1):  # from NL to NU
        IR = int(I - NL + 1)
        QI[0], QI[1], QI[2], QI[3] = list(map(float, lb[I - 1][0:16].split()))
        IRD = int(QI[0] + QI[1] + 0.5)
        for J in range(I, NU + 1):
            IC = int(J - NL + 1)
            QJ[0], QJ[1], QJ[2], QJ[3] = list(map(float, lb[J - 1][0:16].split()))
            ICD = int(QJ[0] + QJ[1] + 0.5)
            H[IR - 1, IC - 1] = 0.0

            # Spin orbit interaction
            if I == J:
                if QI[0] == 0:
                    H[IR - 1, IR - 1] = P[5]
                elif QI[0] == 1:
                    H[IR - 1, IR - 1] = P[0] + 0.5 * P[1] * (QI[1] * (QI[1] + 1) - 2.75)

            # Electronic Zeeman interaction
            W[0] =THJ(float(QI[1]), 1.0,  float(QJ[1]), float(-QI[2]), float(QI[2]-QJ[2]), float(QJ[2]))

            if QI[3] == QJ[3]:
                N = QI[2] - 0.5
                W[2] = W[0]
                if N % 2 != 0:
                    W[2] = -W[2]
                H[IR - 1, IC - 1] += P[11] * W[2] * R[IRD - 1, ICD - 1, 0]

            # Nuclear hyperfine interaction
            W[1] = THJ(float(QI[4]), 1.0, float(QI[4]), float(-QI[3]), float(QI[3] - QJ[3]), float(QJ[3]))
            W[2] = W[0] * W[1] * W[3]
            N = QI[4] - QI[3] - QJ[2] + 0.5
            if N % 2 != 0:
                W[2] = -W[2]
            W[2] *= R[IRD - 1, ICD - 1, 1]

            if QI[0] == 0:
                H[IR - 1, IC - 1] += P[6] * W[2]
            if QI[0] == 1 and QI[1] == QJ[1] and QI[1] == 1.5:
                H[IR - 1, IC - 1] += (P[2] + P[3]) * W[2]
            if QI[0] == 1 and QI[1] == QJ[1] and QI[1] == 0.5:
                H[IR - 1, IC - 1] += (P[2] - P[3]) * W[2]
            if QI[0] == 1 and QI[1] != QJ[1]:
                H[IR - 1, IC - 1] += P[2] * W[2]

            # Electric quadrupole interaction
            W[0] = THJ(float(QI[1]), 2.0, float(QJ[1]), float(-QI[2]), float(QI[2] - QJ[2]), float(QJ[2]))
            W[2] = THJ(float(QI[4]), 2.0, float(QI[4]), float(-QI[3]), float(QI[3] - QJ[3]), float(QJ[3]))
            W[0] = W[0]* W[2] * W[4]
            N = QI[4] - QI[3] - QJ[2] - 0.5
            if N % 2 != 0:
                W[0] = -W[0]
            H[IR - 1, IC - 1] += P[4] * W[0] * R[IRD - 1, ICD - 1, 2]

            # Nuclear Zeeman interaction
            if QI[1] == QJ[1] or QI[2] == QJ[2]:
                W[0] = W[1] * W[3] / P[8]
                N = QI[4] - QI[3] + 1
                if N % 2 != 0:
                    W[0] = -W[0]
                H[IR - 1, IC - 1] += P[7] * P[11] * BN * W[0]

            # Set lower triangular half equal to the upper triangular half
            if I != J:
                H[IC - 1, IR - 1] = H[IR - 1, IC - 1]

    return NH, H


def label(S):
    # Initialize variables
    global nlb, lb
    nlb=0
    # S is nuclear spin
    A =[1.0, 1.0, 0.0]
    B =[1.5, 0.5, 0.5]


    # NP and NS states in order of decreasing J and decreasing JM + IM
    N = int(2 * S + 1)
    P = S + 1.5
    NN = int(2 * P + 1)

    # Loop through NN
    for I in range(NN):
        for J in range(3):
            M = int(2 * B[J] + 1)
            for K in range(M):
                JM = B[J] - K
                for L in range(N):
                    IM = S - L
                    if (JM + IM) == P:
                        # label the value of l, J, Mj, Mi, and Mj+Mi
                        lb[nlb] = f"{A[J]:.1f} {B[J]:.1f} {JM:.1f} {IM:.1f} {(JM + IM):.1f}"
                        nlb += 1
        P=P-1



def main():
    FAR, LF = 4.8436836e-10, 3.541129e-12  # Constants
    C = np.zeros((3, 7))  # Matrix C initialized
    P = np.zeros(17)  # Array P initialized
    Z = np.zeros(6)  # Array Z initialized
    ND = np.zeros(8, dtype=int)  # Array ND initialized
    global lb, nlb
    lb= [''] * 136  # Placeholder for LB character array
    nlb = 0

    ATOM = "NA"
    JS=0 # JS is the index of the isotopes: 0 as the 1st one and 1 as the 2nd one

    # Fill in parameter array for first isotope
    P = atomic_constant(ATOM, P, JS, 1)  # Call ATCTS function (atomic constants)

    IS = JS  # ORNL code try to get two isotopes of one element

    Bmag=16  # in kGauss
    P[11]=Bmag

    # set up a scratch array for transition frequency and F-valves
    scratch=np.zeros((50, 4))

    # Eigenvalue and eigenvector calculation for each isotope
    [P, ND, scratch] = level(P, ND[IS], scratch)  # Call LEVEL function
    # Process the results
    # Update C and other calculations
    C[IS, 2] = P[14] + P[15]
    C[IS, 6] = P[16]
    # P=atomic_constant(ATOM, P, JS, 0)  # Call atomic_constant function again for second isotope
    # If JS!=0: IS=JS and go back to call level function

    print(f"How many transitions: {ND}")
    print(scratch)
    # Adjust F-values for circularly polarized light
    for i in range(IS):
        C[i, 0] = 0.0
        C[i, 1] = 0.0
        for j in range(ND):
            Z[0] = scratch[j, 0]
            Z[1] = scratch[j, 1]
            Z[2] = scratch[j, 2]
            Z[3] = scratch[j, 3] # Ruohong changed

            if Z[2] == -1.0:
                C[i, 0] += Z[1]
            if Z[2] == 1.0:
                C[i, 1] += Z[1]
        C[i, 0] = C[i, 3] / C[i, 0]
        C[i, 1] = C[i, 3] / C[i, 1]

    # input frequency
    FQ = 16952  # Read frequency in cm-1


    # Transition calculations
    for i in range(IS):
        C[i, 3] = 0.0
        C[i, 4] = 0.0
        C[i, 5] = 0.0
        if P[11] != 0.0:
            for j in range(ND):
                Z[0] = scratch[j, 0]
                Z[1] = scratch[j, 1]
                Z[2] = scratch[j, 2]
                Z[3] = scratch[j, 3]  # Ruohong changed
                Z[4] = (LF * Z[0] * Z[0] * Z[1]) ** 2
                if Z[2] == -1.0:
                    Z[1] = C[i, 0] * Z[1]
                if Z[2] == 1.0:
                    Z[1] = C[i, 1] * Z[1]
                Z[1] = Z[0] * Z[1] * Z[2]
                Z[2] = (Z[0] - FQ) * (Z[0] + FQ)
                Z[5] = 2.0 * Z[1] * FQ
                Z[1] = (Z[1] * Z[2]) / (Z[2] * Z[2] + Z[4] * FQ * FQ)
                C[i, 3] -= Z[1]
                C[i, 5] -= Z[1] * Z[3]
                Z[2] = Z[2] * Z[2]
                Z[5] = Z[5] * (Z[2] - Z[0] * Z[0] * Z[4]) / (Z[2] + Z[4] * FQ * FQ) ** 2
                C[i, 4] -= Z[5]
            C[i, 5] = C[i, 5] / C[i, 3]



    # Calculate effective Faraday and alpha coefficients
    Z[0] = 0.0
    Z[1] = 0.0
    Z[2] = 0.0
    if P[11] != 0.0:
        for i in range(IS):
            Z[0] += C[i, 3] * C[i, 6]
            Z[1] += C[i, 4] * C[i, 6]
            Z[2] += C[i, 3] * C[i, 5] * C[i, 6]
            Z[3]=Z[3]/Z[1]
    else:
        Z[2] = 1e-38  # Adjust as per original code

    Z[0] *= FAR
    Z[1] *= FAR
    print(f"Frequency: {FQ}, Faraday: {Z[0] / 60.0}, Alpha: {Z[1] / 60.0}, Coeff: {Z[2]}")


if __name__ == "__main__":
    main()








