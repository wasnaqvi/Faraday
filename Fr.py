# This is a program to calculate the Faraday Polarization Rotation. We will focuse o the Verdet Constant of Rb vapor.
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy 
from sympy.physics.wigner import wigner_3j


def ATCTS_naqvi(Atom, p, n, start):
    '''
    Atom: Character string designating the atomic symbol
    P: Real Linear array to store atomic constants upon return
    N: Integer equal to the index of the isotope to be used
    start: Integer equal to 1 if the first time the function is called
    '''
    # Create a 12x8 zero matrix
    C = np.zeros((12, 8))
    
    # Fill columns with atomic data
    C[0:12, 0] = [14903.89, 0.2235559, 1.6918e-4, -1.2882e-4, -3.686e-7, 0, 0.0050707, 0.82203, 1, 26.4, 26.4, 0.742]
    C[0:12, 1] = [14903.89, 0.2235559, 4.467e-4, -3.4017e-4, -1.843e-5, 0, 0.0134, 3.256, 1.5, 26.4, 26.4, 0.9258]
    C[0:12, 2] = [16967.647, 11.464, 1.177e-3, -5.837e-6, 2.35e-4, 0, 2.95475e-2, 2.217, 1.5, 16, 16, 1]
    C[0:12, 3] = [13023.65, 38.48, 3.642e-4, 8.895e-6, 2.31e-4, 0, 7.70065e-3, 0.3914, 1.5, 26, 26, 0.932]
    C[0:12, 4] = [13023.65, 38.48, 2.004e-4, 4.894e-6, 2.82e-4, 0, 4.236e-3, 0.21487, 1.5, 26, 26, 0.0688]
    C[0:12, 5] = [12737.36, 158.4, 1.525e-3, 2.496e-5, 2.0864e-3, 0, 0.03375, 1.3524, 2.5, 25.5, 28.1, 0.7215]
    C[0:12, 6] = [12737.36, 158.4, 5.149e-3, 8.428e-5, 1.0432e-3, 0, 0.11399, 2.75, 1.5, 25.5, 28.1, 0.2785]
    C[0:12, 7] = [11547.65, 369.41, 3.5692e-3, -2.254e-4, -3.2e-5, 0, 0.07665, 2.57790, 3.5, 29.9, 34.0, 1]

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
                k = min(i + n - 1, 7)  # For subsequent isotopes

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

Atom = 'LI'  # Lithium
p = np.zeros(17)  # Array to store results
n = 2 # Select the second isotope (Li007)
start = 1  # First time calling

# Call the function for Li007
p_result_li007 = ATCTS_naqvi(Atom, p, n, start)
print("Results for Li007:")
print(p_result_li007)


def lgn(n):
    '''
    The purpose of this function is to compute the natural logarithm of the factorial of n.
    
    Parameter(s):
    n: Integer > 0.
    '''
    
    if n <0:
        raise ValueError("The input must be a positive integer.")
    
    return math.lgamma(n+1)

def lgn_stirling(n):
    if n <= 0:
        raise ValueError("n must be a positive integer greater than 0.")    
    # Stirling's approximation formula: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2*pi*n)
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
'''
# Example with large n
n_large = 10000
result_stirling = lgn_stirling(n_large)
result_gamma = lgn(n_large)
print(f"ln({n_large}!) using Stirling's approximation = {result_stirling}")
print(f"ln({n_large}!) using Gamma approximation = {result_gamma}")
'''

def THJ(J1, J2, J3, M1, M2, M3):
    '''
    The purpose of this function is to compute the 3j symbol.
    
    Parameters:
    J1, J2, J3: Integers or half-integers.
    M1, M2, M3: Integers or half-integers.
    '''
    
    if (J1 < 0) or (J2 < 0) or (J3 < 0) or (M1 < 0) or (M2 < 0) or (M3 < 0):
        raise ValueError("The input must be a positive integer or half-integer.")
    
    return float(wigner_3j(J1, J2, J3, M1, M2, M3))

print(THJ(2, 6, 4, 0, 0, 0))

def Matrix(nh,H,p,nl,nu):
    '''
    The purpose of this function is to compute the matrix elements of the Atomic Hamiltonian.
    
    Parameters:
    nh: Declared row dimension of H.
    H: Matrix to store the results upon return.
    P: Linear array containing parameters values(atomic constants).
    NL: Integer equal to the lower position of Array LB for block MJ+MI.
    NU: Integer equal to the upper position of Array LB for block MJ+MI.
    '''
    W = np.zeros(5)
    BN = 2.542e-5
    QI=np.zeros(5)
    QJ=np.zeros(4)
    R = np.array([
    [1.14490551520957E-01, 0.00000000000000E+00, 0.00000000000000E+00],
    [0.00000000000000E+00, 5.7735026918962577E-01, -5.43383832291511E-02],
    [-2.40202829036970E-01, 0.00000000000000E+00, 1.20474871391589E-02],
    [5.7735026918962577E-01, 3.2655980232371094E-01, 5.7735026918962577E-01],
    [-2.412275077709000E-01, 1.020448713915E-02, -5.403883822915E-02]])
  
    NLB = 0  
    LB = [""] * 136  

    
    W[3] = np.sqrt(0.2 * p[8] * (p[8] + 1) * (2.0 * p[8] + 1.0))
    W[4] *= (2.0 * p[8] + 3.0) / (p[8] * (2.0 * p[8] - 1.0))

    ''' Calculating the Hamiltonian Matrix For Block MJ+MI '''
    
    
    
    for I in range(nu, nl+1):
         IR = I - nl + 1

    # Simulating DECODE(16, 2, LB(I))(1:16) - Split and parse string
         QI = [float(LB[I][j:j+4]) for j in range(0, 16, 4)]
    
         IRD = QI[0] + QI[1] + 0.5

    # Loop over J from NL to NU
         for J in range(nl, nu+1):
               IC = J - nl + 1
        # Simulating DECODE(16, 2, LB(J))(1:16) - Split and parse string
               QJ = [float(LB[J][j:j+4]) for j in range(0, 16, 4)]
               ICD=QJ[0]+QJ[1]+0.5
        
               H[IR-1, IC-1] = 0.0
               
    # Spin Orbit Interaction
    if (I!=J):
         #  Electronic Zeeman
        W[0]=THJ(QI[0], 1.0, QJ[1], -QI[2],QI[2]-QJ[2], QJ[2])  # Wigner 3j symbol.
    if(QI[3] != QJ[3]):
            W[1] = THJ(QI[4], 1.0, QI[4], -QI[4], QI[4] - QJ[4], QJ[4]) * np.sqrt(2.0)
    if QI[0]==0 : 
        H[IR-1, IR-1]=p[5]
    if QI[0]==1:
        H[IR-1, IR-1]=p[0]         
    N=QI[2]-0.5
    W[2]=W[0]
    if(N%2 != 0):
        W[2] = -W[2]
    H[IR-1, IC-1] = H[IR-1, IC-1] + p[12] * W[2] * R[IRD,ICD-1, 1]  
    
    # Nuclear Hyperfine Interaction
    W[2]=W[0]*W[1]*W[3]
    N=QI[4]-QI[3]-QJ[1]+0.5
    if(N%2 != 0):
        W[2] = -W[2]
    W[2] = W[2] * R[IRD-1, ICD-1, 1]
    if QI ==0:
        H[IR-1, IC-1] += p[6] * W[2]
    if QI[0] == 1 and QI[1] == QJ[1]:
        H[IR-1, IC-1] += p[2]+2*(B-1)*p[3]*W[2] # What is B? 
    if QI[0] == 1 and QI[1] != QJ[1]:
        H[IR-1, IC-1] += p[3]*W[2]
    
    # Electric Quadrupole section
    W[0] = THJ(QI[1], 2.0, QJ[1], -QI[2], QI[2] - QJ[2], QJ[2])
    W[2] = THJ(QI[4], 2.0, QJ[4], -QI[3], QI[3] - QJ[3], QJ[3])
    W[0] *= W[2] * W[3]
    N=QI[4]-QI[3]-QJ[1]+0.5
    if(N%2 != 0):
        W[0] = -W[0]
    H[IR-1, IC-1] += p[4] * W[0] * R[IRD-1, ICD-1, 2]
    
    # Nuclear Zeeman Interaction
    
    # Nuclear Zeeman section
    if QI[1] != QJ[1] or QI[2] != QJ[2]:
            # Set lower triangular half equal to upper triangular half
            if I != J:
                 H[IC-1, IR-1] = H[IR-1, IC-1]

    W[0] = W[1] * W[3] / p[8]
    N = QI[4] - QI[3] + 1
    if int(N) % 2 != 0:
         W[0] = -W[0]

    H[IR-1, IC-1] += p[7] + p[11] * BN * W[0]
    
    return H

    
def Matrix2(nh, H, p, nl, nu):
    '''
    The purpose of this function is to compute the matrix elements of the Atomic Hamiltonian.
    
    Parameters:
    nh: Declared row dimension of H.
    H: Matrix to store the results upon return.
    P: Linear array containing parameters values (atomic constants).
    NL: Integer equal to the lower position of Array LB for block MJ+MI.
    NU: Integer equal to the upper position of Array LB for block MJ+MI.
    '''
    W = np.zeros(5)
    BN = 2.542e-5
    QI = np.zeros(5)
    QJ = np.zeros(4)
    R = np.array([
        [1.14490551520957E-01, 0.00000000000000E+00, 0.00000000000000E+00],
        [0.00000000000000E+00, 5.7735026918962577E-01, -5.43383832291511E-02],
        [-2.40202829036970E-01, 0.00000000000000E+00, 1.20474871391589E-02],
        [5.7735026918962577E-01, 3.2655980232371094E-01, 5.7735026918962577E-01],
        [-2.412275077709000E-01, 1.020448713915E-02, -5.403883822915E-02]
    ])

    NLB = 0  
    LB = [""] * 136  

    W[3] = np.sqrt(0.2 * p[8] * (p[8] + 1) * (2.0 * p[8] + 1.0))
    W[4] = (2.0 * p[8] + 3.0) / (p[8] * (2.0 * p[8] - 1.0))

    ''' Calculating the Hamiltonian Matrix For Block MJ+MI '''
    
    for I in range(nl, nu+1):
        IR = I - nl + 1

        # Simulating DECODE(16, 2, LB(I))(1:16) - Split and parse string
        QI = [float(LB[I][j:j+4]) for j in range(0, 16, 4)]
        IRD = QI[0] + QI[1] + 0.5

        for J in range(nl, nu+1):
            IC = J - nl + 1
            QJ = [float(LB[J][j:j+4]) for j in range(0, 16, 4)]
            ICD = QJ[0] + QJ[1] + 0.5
        
            H[IR-1, IC-1] = 0.0

            if I != J:
                # Spin-Orbit Interaction and Electronic Zeeman
                W[0] = THJ(QI[0], 1.0, QJ[1], -QI[2], QI[2] - QJ[2], QJ[2])
                if QI[3] != QJ[3]:
                    W[1] = THJ(QI[4], 1.0, QI[4], -QI[4], QI[4] - QJ[4], QJ[4]) * np.sqrt(2.0)

                N = QI[2] - 0.5
                W[2] = W[0]
                if int(N) % 2 != 0:
                    W[2] = -W[2]
                H[IR-1, IC-1] += p[12] * W[2] * R[IRD-1, ICD-1, 1]

                # Nuclear Hyperfine Interaction
                W[2] = W[0] * W[1] * W[3]
                N = QI[4] - QI[3] - QJ[1] + 0.5
                if int(N) % 2 != 0:
                    W[2] = -W[2]
                H[IR-1, IC-1] += p[6] * W[2] * R[IRD-1, ICD-1, 1]

                if QI[0] == 0:
                    H[IR-1, IC-1] += p[6] * W[2]
                elif QI[0] == 1 and QI[1] == QJ[1]:
                    H[IR-1, IC-1] += p[2] + 2 * (B - 1) * p[3] * W[2]  # B needs to be defined
                elif QI[0] == 1 and QI[1] != QJ[1]:
                    H[IR-1, IC-1] += p[3] * W[2]

                # Electric Quadrupole section
                W[0] = THJ(QI[1], 2.0, QJ[1], -QI[2], QI[2] - QJ[2], QJ[2])
                W[2] = THJ(QI[4], 2.0, QJ[4], -QI[3], QI[3] - QJ[3], QJ[3])
                W[0] *= W[2] * W[3]
                H[IR-1, IC-1] += p[4] * W[0] * R[IRD-1, ICD-1, 2]

                # Nuclear Zeeman Interaction
                if QI[1] != QJ[1] or QI[2] != QJ[2]:
                    H[IC-1, IR-1] = H[IR-1, IC-1]

                W[0] = W[1] * W[3] / p[8]
                N = QI[4] - QI[3] + 1
                if int(N) % 2 != 0:
                    W[0] = -W[0]

                H[IR-1, IC-1] += p[7] + p[11] * BN * W[0]

    return H
   
def test_Matrix():
    # Example small matrix H (3x3) to test the function
    nh = 3
    H = np.zeros((nh, nh))

    # Sample atomic constants (randomized for testing, modify as needed)
    p = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 
        10.0, 11.0, 12.0, 13.0
    ]

    # Set the lower and upper bounds
    nl = 1
    nu = 3

    # Call the Matrix function with the test inputs
    H_result = Matrix(nh, H, p, nl, nu)

    # Print the result for inspection
    print("Resulting Matrix H:")
    print(H_result)

    # You can also add assertions for automated testing, for example:
    assert H_result.shape == (nh, nh), "Matrix dimensions do not match"
    assert np.all(H_result >= 0), "Matrix should not contain negative values in this test"
    
    print("Test passed!")


# Run the test function
test_Matrix()

   
            
            
        

