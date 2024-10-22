def espectro(A):
    return [abs(x) for x in A.eigenvalues()]

def radio_espectral(A):
    return max(espectro(A))

def converge(A) -> bool:
    return radio_espectral(A) < 1

def separar_matriz(A):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    n = A.dimensions()[0]
    D = diagonal_matrix(QQ, [A[i,i] for i in n])
    E = matrix(QQ, [[0 if i<=j else -A[i,j] for j in range(n)] for i in range(n)])
    F = matrix(QQ, [[0 if i>=j else -A[i,j] for j in range(n)] for i in range(n)])
    return D,E,F

def jacobi_matriz(A):
    D,E,F = separar_matriz(A)
    C = ~D
    J = C*(E+F)
    return J, C

def gauss_seidel_matriz(A):
    D,E,F = separar_matriz(A)
    C = ~(D-E)
    L1 = C*F
    return L1, C

def sor_matriz(A,w):
    D,E,F = separar_matriz(A)
    M = (w^(-1))*D - E
    N = F + (w^(-1) - 1)*D
    C = ~M
    Lw = C*N
    return Lw, C

def sor_wopt(A):
    if not A.is_hermitian() or not A.is_positive_definite():
        raise ValueError("La matriz no es hermitiana definida positiva")
    n = A.dimensions()[0]
    for i in range(n):
        for j in range(n):
            if (j < i - 1 or j > i + 1) and A[i, j] != 0:
                raise ValueError("La matriz no es tridiagonal")
    J = jacobi_matriz(A)
    k = radio_espectral(J)
    return 2/(1+sqrt(1-k^2))

def jacobi_punto(A, b, x0, tol=10^-3, kmax=20, pausa=0, v=True):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    if A.is_singular():
        raise ValueError("La matriz del sistema es singular")
    
    n = A.dimensions()[0]
    if n != len(b.list()):
        raise ValueError("Las dimensiones no concuerdan")
        
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("Un elemento diagonal es nulo")
    
    if v: print("Matriz original:")
    if v: print(A)
    if v: print("Término independiente:")
    if v: print(b)
    time.sleep(pausa)
    if v: print("Comenzamos el método de Jacobi")

    x = x0.copy()
    error = tol + 1
    k = 0

    while error > tol and k < kmax:
        error = 0
        y = x.copy()

        for i in range(n):
            x[i] = b[i]
            for j in range(i):
                x[i] -= A[i, j] * y[j]
            for j in range(i + 1, n):
                x[i] -= A[i, j] * y[j]
            x[i] /= A[i, i]
            
            error += abs(y[i] - x[i])
        
        k += 1
        if v: print("Resultado de la iteración", k, "del método de Jacobi:")
        if v: print("X =", x)
        if v: print("Error =", error)
        time.sleep(pausa)

    if k >= kmax:
        print("Número máximo de iteraciones superado")
    
    return x, error

def jacobi_matricial(A, b, x0, tol=10^-3, kmax=20, v=True):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
        
    A = matriz.dimensions()[0]
    J, C = jacobi_matriz(A)
    err = tol + 1
    u = x0.copy()
    
    while err > tol and k < kmax:
        v = J*u + C*b
        u = v
        err = (A*v-b).norm()
        if v: print("Resultado de la iteración", k, "del método de Jacobi:")
        if v: print("X =", v)
        if v: print("Error =", err)
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return v,err

def gauss_seidel_punto(A, b, x0, tol=10^-3, kmax=20, pausa=0, v=True):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    if A.is_singular():
        raise ValueError("La matriz del sistema es singular")
    
    n = A.dimensions()[0]
    if n != len(b.list()):
        raise ValueError("Las dimensiones no concuerdan")
        
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("Un elemento diagonal es nulo")
     
    if v: print("Matriz original:")
    if v: print(A)
    if v: print("Término independiente:")
    if v: print(b)
    time.sleep(pausa)
    if v: print("Comenzamos el método de Gauss-Seidel")

    x = x0.copy()
    error = tol + 1
    k = 0

    while error > tol and k < kmax:
        error = 0
        for i in range(n):
            y = x[i]
            x[i] = b[i]
            for j in range(i):
                x[i] -= A[i, j] * x[j]
            for j in range(i + 1, n):
                x[i] -= A[i, j] * x[j]
            x[i] /= A[i, i]
            
            error += abs(y - x[i])
        
        k += 1
        
        if v: print("Resultado de la iteración", k, "del método de Gauss-Seidel:")
        if v: print("X =", x)
        if v: print("Error =", error)
        time.sleep(pausa)

    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return x, error

def gauss_seidel_matricial(A, b, x0, tol=10^-3, kmax=20, v=True):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
        
    A = matriz.dimensions()[0]
    L1, C = gauss_seidel_matriz(A)
    err = tol + 1
    u = x0.copy()
    
    while err > tol and k < kmax:
        v = L1*u + C*b
        u = v
        err = (A*v-b).norm()
        if v: print("Resultado de la iteración", k, "del método de Gauss-Seidel:")
        if v: print("X =", v)
        if v: print("Error =", err)
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return v,err

def sor_matricial(A, b, x0, w, tol=10^-3, kmax=20, v=True):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    A = matriz.dimensions()[0]
    Lw, C = sor_matriz(A, w)
    err = tol + 1
    u = x0.copy()
    
    while err > tol and k < kmax:
        v = Lw*u + C*b
        u = v
        err = (A*v-b).norm()
        if v: print("Resultado de la iteración", k, "del método de sobrerrelajación:")
        if v: print("X =", v)
        if v: print("Error =", err)
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return v,err

def iteraciones_tol(metodo, b, x0, tol, norma=Infinity):
    M, C = metodo
    v = M*x0 + C*b
    k = log((1 - tol*M.norm(norma))/((v-x0).norm(norma)),M.norm(norma)) + 1
    return ceil(k)

def lu_doolittle(A):
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    if A.is_singular():
        raise ValueError("La matriz del sistema es singular")
    
    P,L,U = A.LU()
    return ~P,L,U

def lu_crout(A):
    P,L,U = lu_doolittle(A)
    n = A.dimensions()[0]
    M = zero_matrix(QQ,n,n)
    N = zero_matrix(QQ,n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = U[i,j]/U[i,i]
            N[i,j] = L[i,j]*U[j,j]
    U = M
    L = N
    return P,L,U

def lu_cholesky(A):
    P,L,U = lu_doolittle(A)
    n = A.dimensions()[0]
    M = zero_matrix(QQ,n,n)
    N = zero_matrix(QQ,n,n)
    for i in range(n):
        for j in range(n):
            M[i,j] = U[i,j]/sqrt(U[i,i])
            N[i,j] = L[i,j]*sqrt(U[j,j])
    U = M
    L = N
    return P,L,U
