# This is a program to calculate the Faraday Polarization Rotation. We will focuse o the Verdet Constant of Rb vapor.
import numpy as np
import matplotlib.pyplot as plt


def  ATCTS_schulick(Atom,p,n,start):
    '''
    Atom character string designating th atomic symbol
    P Real Lineary array to store atomic constants upon return
    N Integer Equal to the Index of Isotope to be used
    start Integer Equal to 1 if the first time the function is called. False for calls thereafter for the same atom
    '''
     # Create a 12x8 zero matrix
    C = np.zeros((12, 8))
    
    # Fill column 1 (index 0 in Python)
    C[0:12, 0] = [14903.89, 0.2235559, 1.6918e-4, -1.2882e-4, -3.686e-7, 0, 0.0050707, 0.82203, 1, 26.4, 26.4, 0.742]
    C[0:12, 1] = [14903.89, 0.2235559, 4.467e-4, -3.4017e-4, -1.843e-5, 0, 0.0134, 3.256, 1.5, 26.4, 26.4, 0.9258]
    C[0:12, 2] = [16967.647, 11.464, 1.177e-3, -5.837e-6, 2.35e-4, 0, 2.95475e-2, 2.217, 1.5, 16, 16, 1]
    C[0:12, 3] = [13023.65, 38.48, 3.642e-4, 8.895e-6, 2.31e-4, 0, 7.70065e-3, 0.3914, 1.5, 26, 26, 0.932]
    C[0:12, 4] = [13023.65, 38.48, 2.004e-4, 4.894e-6, 2.82e-4, 0, 4.236e-3, 0.21487, 1.5, 26, 26, 0.0688]
    C[0:12, 5] = [12737.36, 158.4, 1.525e-3, 2.496e-5, 2.0864e-3, 0, 0.03375, 1.3524, 2.5, 25.5, 28.1, 0.7215]
    C[0:12, 6] = [12737.36, 158.4, 5.149e-3, 8.428e-5, 1.0432e-3, 0, 0.11399, 2.75, 1.5, 25.5, 28.1, 0.2785]
    C[0:12, 7] = [11547.65, 369.41, 3.5692e-3, -2.254e-4, -3.2e-5, 0, 0.07665, 2.57790, 3.5, 29.9, 34.0, 1]

    FT=1.499e9
    Label=['LI006','LI007','NA023','K039','K041','RB085','RB087','CS133']
      
    j,k=0,0
    for i in range(0,8):
        if Atom!=Label[i][0]:
            continue
        j+=1
        k=i
    
    k-=j
    if n<j:
        n+=1
    k+=n
    for i in range(11):
        p[i]=C[i,k]
    
    p[12] = np.sqrt(FT / (p[0] - p[1]) ** 3 / p[10])
    p[13] = np.sqrt(2.0 * FT / (p[0] + 0.5 * p[1]) ** 3 / p[9])
    p[14] = FT / (p[0] - p[1]) ** 2 / p[10]
    p[15] = 2.0 * FT / (p[0] + 0.5 * p[1]) ** 2 / p[9]
    p[16] = C[11, k]
    
    print("ATCTS was called")
    return p

def ATCTS_2(Atom, p, n, start):
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

    # Reset n if this is the first call
    if start == 1:
        n = 0

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


# Initialize variables for Rubidium (Rb)
Atom = 'RB'  # Focus on Rubidium
p = np.zeros(17)  # Array to store results, size 17 since p[0] to p[16] are used
n = 1 # Index for Rb085 (set to 1 for the first isotope)
start = 0  # First call

# Call the function
p_result = ATCTS_2(Atom, p, n, start)

# Output the result
print("Computed Constants (p array):")
print(p_result)
