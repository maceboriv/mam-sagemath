import time

def espectro(A):
    """Devuelve el espectro de la matriz A en forma de lista"""
    return A.eigenvalues()

def radio_espectral(A):
    """Devuelve el radio espectral de la matriz A, es decir, el mayor VAP en valor absoluto"""
    return max(map(abs, espectro(A)))

def converge(A) -> bool:
    """Dada una matriz A (asociada a un método) devuelve `True` si el método converge o False en caso contrario

    Es equivalente decir que:
        1. p(A) < 1
        2. existe al menos una norma matricial tal que ||A|| < 1
    """
    return radio_espectral(A) < 1

def separar_matriz(A):
    """Dada una matriz cuadrada A, devuelve su descomposición (D, E, F) tal que A = D - E - F."""
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    n = A.dimensions()[0]
    D = diagonal_matrix(QQ, [A[i,i] for i in range(n)])
    E = matrix(QQ, [[0 if i<=j else -A[i,j] for j in range(n)] for i in range(n)])
    F = matrix(QQ, [[0 if i>=j else -A[i,j] for j in range(n)] for i in range(n)])
    return D,E,F

def jacobi_matriz(A):
    """Calcula J y C para el método de Jacobi, siendo A la matriz del sistema. Devuelve así una tupla de dos matrices.
    
    Ejemplo:

    |    J, C = jacobi_matriz(A)
    |    x1 = J*x0 + C*b
    """
    D,E,F = separar_matriz(A)
    C = ~D
    J = C*(E+F)
    return J, C

def gauss_seidel_matriz(A):
    """Calcula L1 y C para el método de Gauss-Seidel, siendo A la matriz del sistema. Devuelve así una tupla de dos matrices.
    
    Ejemplo:

    |    L1, C = gauss_seidel_matriz(A)
    |    x1 = L1*x0 + C*b
    """
    D,E,F = separar_matriz(A)
    C = ~(D-E)
    L1 = C*F
    return L1, C

def sor_matriz(A,w):
    """ Dada A matriz de un sistema de ecuaciones y w el valor de relajación, calcula Lw y C. Devuelve así una tupla de dos matrices.

    Este método converge más rápido que Jacobi.

    Ejemplo:

    |    Lw, C = sor_matriz(A)
    |    x1 = Lw*x0 + C*b
    """
    D,E,F = separar_matriz(A)
    M = (w^(-1))*D - E
    N = F + (w^(-1) - 1)*D
    C = ~M
    Lw = C*N
    return Lw, C

def sor_wopt(A):
    """Dada A una matriz hermitiana, definida positiva y tridiagonal devuelve el valor óptimo para el método de sobrerrelajación, que se calcula de forma analítica y como se ha visto en clase.

    Hay que tener en cuenta lo siguinte:
        1. Si A es estrictamente diagonal dominante, Jacobi y Gauss-Seidel convergen; SOR converge sii 0 < w <= 1.
        2. Si SOR connverge, p(Lw) >= |w-1|, w != 0
        3. Si A es simétrica y definida positiva, SOR converge sii 0 < w < 2
        4. Si A es hermitiana, definida positiva y tridiagonal, p(Lw) = { 1/4 (w*p(J) + sqrt((w*p(J))^2 - 4*(w-1))) si 0 < w < wopt // w-1 si wopt < w < 2 }
    """
    if not A.is_hermitian() or not A.is_positive_definite():
        raise ValueError("La matriz no es hermitiana definida positiva")
    n = A.dimensions()[0]
    for i in range(n):
        for j in range(n):
            if (j < i - 1 or j > i + 1) and A[i, j] != 0:
                raise ValueError("La matriz no es tridiagonal")
    J = jacobi_matriz(A)[0]
    k = radio_espectral(J)
    return 2/(1+sqrt(1-k^2))

def jacobi_punto(A, b, x0, tol=10^-3, kmax=20, pausa=0, v=True):
    """Implementación del método de Jacobi punto a punto (superior al matricial)

    Parámetros:
    - A es la matriz del sistema (cuadrada, no singular y con la diagonal no nula)
    - b es el vector de términos independientes
    - x0 es la aproximación inicial (si no se especifica deberías pasar el vector nulo)
    - tol es la tolerancia del error, el método termina si el error es más pequeño que tol
    - kmax es el número máximo de iteraciones que realiza el método antes de parar
    - pausa es un número entero que introduce una pausa de tiempo (en segundos) entre iteración e iteración
    - v es un parámetro que controla los mensajes. Si se pone en True, imprimirá cada paso del método
    
    Ejemplo:

    |    A = matrix(QQ, [[5,-2,3],[-3,9,1],[2,-1,-7]])
    |    b = vector(QQ, [-1,2,3])
    |    x0 = vector(QQ, [1,1,1])
    |    x,err = jacobi_punto(A, b, x0, 10^-2, 0, False)
    |    print(x, err)
    """
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

    x = copy(x0)
    error = tol + 1
    k = 0

    while error > tol and k < kmax:
        error = 0
        y = copy(x)

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
        if v: print("X =", N(x))
        if v: print("Error =", N(error))
        time.sleep(pausa)

    if k >= kmax:
        print("Número máximo de iteraciones superado")
    
    return N(x), N(error)

def jacobi_matricial(A, b, x0, tol=10^-3, kmax=20, v=True):
    """Implementación del método de Jacobi de forma matricial (menos eficiente que el punto a punto)

    Parámetros:
    - A es la matriz del sistema (cuadrada)
    - b es el vector de términos independientes
    - x0 es la aproximación inicial (si no se especifica deberías pasar el vector nulo)
    - tol es la tolerancia del error, el método termina si el error es más pequeño que tol
    - kmax es el número máximo de iteraciones que realiza el método antes de parar
    - pausa es un número entero que introduce una pausa de tiempo (en segundos) entre iteración e iteración
    - v es un parámetro que controla los mensajes. Si se pone en True, imprimirá cada paso del método
    
    Ejemplo:

    |    A = matrix(QQ, [[5,-2,3],[-3,9,1],[2,-1,-7]])
    |    b = vector(QQ, [-1,2,3])
    |    x0 = vector(QQ, [1,1,1])
    |    x,err = jacobi_matricial(A, b, x0, 10^-2, 0, False)
    |    print(x, err)
    """
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    J, C = jacobi_matriz(A)
    err = tol + 1
    u = copy(x0)
    k = 0

    while err > tol and k < kmax:
        v = J*u + C*b
        u = v
        err = (A*v-b).norm()
        k += 1
        if v: print("Resultado de la iteración", k, "del método de Jacobi:")
        if v: print("X =", N(v))
        if v: print("Error =", N(err))
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return N(v),N(err)

def gauss_seidel_punto(A, b, x0, tol=10^-3, kmax=20, pausa=0, v=True):
    """Implementación del método de Gauss-Seidel punto a punto (superior al matricial)

    Parámetros:
    - A es la matriz del sistema (cuadrada, no singular y con la diagonal no nula)
    - b es el vector de términos independientes
    - x0 es la aproximación inicial (si no se especifica deberías pasar el vector nulo)
    - tol es la tolerancia del error, el método termina si el error es más pequeño que tol
    - kmax es el número máximo de iteraciones que realiza el método antes de parar
    - pausa es un número entero que introduce una pausa de tiempo (en segundos) entre iteración e iteración
    - v es un parámetro que controla los mensajes. Si se pone en True, imprimirá cada paso del método
    
    Ejemplo:

    |    A = matrix(QQ, [[5,-2,3],[-3,9,1],[2,-1,-7]])
    |    b = vector(QQ, [-1,2,3])
    |    x0 = vector(QQ, [1,1,1])
    |    x,err = gauss_seidel_punto(A, b, x0, 10^-2, 0, False)
    |    print(x, err)
    """
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

    x = copy(x0)
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
        if v: print("X =", N(x))
        if v: print("Error =", N(error))
        time.sleep(pausa)

    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return N(x), N(error)

def gauss_seidel_matricial(A, b, x0, tol=10^-3, kmax=20, v=True):
    """Implementación del método de Gauss-Seidel de forma matricial (menos eficiente que el punto a punto)

    Parámetros:
    - A es la matriz del sistema (cuadrada)
    - b es el vector de términos independientes
    - x0 es la aproximación inicial (si no se especifica deberías pasar el vector nulo)
    - tol es la tolerancia del error, el método termina si el error es más pequeño que tol
    - kmax es el número máximo de iteraciones que realiza el método antes de parar
    - pausa es un número entero que introduce una pausa de tiempo (en segundos) entre iteración e iteración
    - v es un parámetro que controla los mensajes. Si se pone en True, imprimirá cada paso del método
    
    Ejemplo:

    |    A = matrix(QQ, [[5,-2,3],[-3,9,1],[2,-1,-7]])
    |    b = vector(QQ, [-1,2,3])
    |    x0 = vector(QQ, [1,1,1])
    |    x,err = gauss_seidel_matricial(A, b, x0, 10^-2, 0, False)
    |    print(x, err)
    """
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    L1, C = gauss_seidel_matriz(A)
    err = tol + 1
    u = copy(x0)
    k = 0
    
    while err > tol and k < kmax:
        v = L1*u + C*b
        u = v
        err = (A*v-b).norm()
        k += 1
        
        if v: print("Resultado de la iteración", k, "del método de Gauss-Seidel:")
        if v: print("X =", N(v))
        if v: print("Error =", N(err))
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return N(v),N(err)

def sor_matricial(A, b, x0, w, tol=10^-3, kmax=20, v=True):
    """Implementación del método de Sobrerrelajación Sucesiva de forma matricial

    Es un método más general al de Gauss-Seidel y con una convergencia más rápida

    Parámetros:
    - A es la matriz del sistema (cuadrada)
    - b es el vector de términos independientes
    - x0 es la aproximación inicial (si no se especifica deberías pasar el vector nulo)
    - w es el parámetro de relajación
    - tol es la tolerancia del error, el método termina si el error es más pequeño que tol
    - kmax es el número máximo de iteraciones que realiza el método antes de parar
    - pausa es un número entero que introduce una pausa de tiempo (en segundos) entre iteración e iteración
    - v es un parámetro que controla los mensajes. Si se pone en True, imprimirá cada paso del método
    
    Ejemplo:

    |    A = matrix(QQ, [[2,1,0,0],[1,2,1,0],[0,1,2,1],[0,0,1,2]])
    |    b = vector(QQ, [1,2,3,4])
    |    x0 = vector(QQ, [0,0,0,0])
    |    wopt = sor_wopt(A)
    |    x, err = sor_matricial(A, b, x0, wopt, 10^-2, 0, False)
    |    print(x, err)
    """
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    Lw, C = sor_matriz(A, w)
    err = tol + 1
    u = copy(x0)
    k = 0

    while err > tol and k < kmax:
        v = Lw*u + C*b
        u = v
        err = (A*v-b).norm()
        k += 1
        
        if v: print("Resultado de la iteración", k, "del método de sobrerrelajación:")
        if v: print("X =", N(v))
        if v: print("Error =", N(err))
    
    if k >= kmax:
        print("Número máximo de iteraciones superado")
        
    return N(v),N(err)

def iteraciones_tol(metodo, b, x0, tol, norma=Infinity):
    """Dada una tupla de dos matrices M,C asociadas a un método iterativo, el término independiente del sistema, una aproximación inicial y una norma matricial, devuelve el número mínimo de iteraciones para que converja."""
    M, C = metodo
    v = M*x0 + C*b
    k = log((1 - M.norm(norma) * tol)/((v-x0).norm(norma)),M.norm(norma))
    return ceil(k)

def lu_doolittle(A):
    """Dada A matriz cuadrada y no singular, devuelve P, L, U tal que PA = LU con L triangular inferior y U triangular superior."""
    if not A.is_square():
        raise ValueError("La matriz no es cuadrada")
    
    if A.is_singular():
        raise ValueError("La matriz del sistema es singular")
    
    P,L,U = A.LU()
    return ~P,L,U

def lu_crout(A):
    """Dada A matriz cuadrada y no singular, devuelve P, L, U tal que PA = LU con L triangular inferior y U triangular superior (intercambian las diagonales)."""
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
    """Dada A matriz cuadrada y no singular (hermitiana y definida positiva para que sea única), devuelve P, L, U tal que PA = L*L^t, con L triangular inferior."""
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

def condicionamiento_matriz(A, norma=Infinity):
    """Devuelve el número de condición de la matriz A asociada a un sistema lineal en la norma dada
    """
    return A.norm(norma)*A.inverse().norm(norma)

# ==================================================================================================================

def punto_fijo(f, x0, niter=20):
    """Aplica niter iteraciones del método del punto fijo a f partiendo de x0
    Recordar que para usar este método es necesario:
        1) f continua en [a,b]
        2) f([a,b]) c [a,b]
    
    1), 2) => Requisitos para que exista al menos un punto fijo (Teorema de Brouwer)
        
        3) f contractiva ó |f'(x)| < 1 en (a,b)
     1),2),3) => Existe un ÚNICO punto fijo (Teorema de Banach)

    Teorema de Schröder: si x* es punto fijo de g y las derivadas n-ésimas evaluadas en x* se anulan hasta la p (siendo esta no nula)
    entonces el método de punto fijo converge como mínimo con orden p.

    Ejemplo:

    |    f(x) = 2*x*(1-x)
    |    punto_fijo(f, 3/4, 15)
    """
    resultados = [x0]

    for i in range(niter):
        x0 = f(x0)
        resultados.append(N(x0))

    return resultados


def biseccion(f, intervalo, tol=10^-3, kmax=10):
    """Aproxima la raíz de f en intervalo con una tolerancia al error tol y kmax iteraciones máximas
    
    Condiciones:
        1) f continua en [a,b]
        2) f(a)f(b)<0

    Cota de error: abs(solucion - c_n) <= 1/(2^(n+1)) * abs(b_0 - a_0)

    Ejemplo:

    |    f(x) = x*sin(x)-1
    |    intervalo = (0, 2)
    |    r, err = biseccion(f, intervalo, 10^-2, 20)
    """
    a, b = intervalo
    a0 = a
    b0 = b
    k = 0
    
    if f(a)*f(b)>=0:
        raise ValueError("f(a)f(b) >= 0") # Comprobamos 2

    while abs(b - a) > tol and k < kmax:
        c = (a + b) / 2             # Sucesión {Cn} de las aproximaciones sucesivas

        if f(a) * f(c) < 0:         # raíz € [a,c]
            b = c
        else:                       # raíz € [c,b]
            a = c
        k += 1

    if k >= kmax:
        print("Número máximo de iteraciones superado")

    raiz = (a + b) / 2
    err = (1/2^(k+1))*abs(b0-a0)    # Cota para el error
    return N(raiz), N(err)

def regula_falsi(f, intervalo, tol=10^-3, kmax=10):
    """Aproxima la raíz de f en intervalo con una tolerancia al error tol y kmax iteraciones máximas
    
    Condiciones:
        1) f continua en [a,b]
        2) f(a)f(b)<0

    Cota de error: abs(solucion - c_n) <= 1/(2^(n+1)) * abs(b_0 - a_0)

    Ejemplo:

    |    f(x) = x*sin(x)-1
    |    intervalo = (0, 2)
    |    r, err = biseccion(f, intervalo, 10^-2, 20)
    """
    
    a, b = intervalo
    a0 = a
    b0 = b
    k = 0
    
    if f(a)*f(b)>=0:
        raise ValueError("f(a)f(b) >= 0") # Comprobamos 2

    while abs(b - a) > tol and k < kmax:
        c = (a*f(b) - b*f(a)) / (f(b)-f(a))      # Sucesión {Cn} de las aproximaciones sucesivas

        if f(a) * f(c) < 0:                      # raíz € [a,c]
            b = c
        else:                                    # raíz € [c,b]
            a = c
        k += 1

    if k >= kmax:
        print("Número máximo de iteraciones superado")

    raiz = (a*f(b) - b*f(a)) / (f(b)-f(a))    
    err = (1/2^(iteracion+1))*abs(b0-a0)         # Cota para el error
    return N(raiz), N(err)

def newton_raphson(f, x0, tol=10^-3, kmax=20):
    """Dada f que cumple las condiciones de convergencia adecuadas, aproxima la raíz en un intervalo (a,b) mediante el método de Newton-Raphson
    
    Condiciones:
        -Convergencia local:
         Existe d tq si (x ± d) ⊆ [a,b] entonces el método converge a x* para todo x ∈ (x ± d)

        -Convergencia global (condiciones de Fourier):
         1) f(a)f(b) < 0
         2) f'(x) != 0 si x∈[a,b]
         3) f''(x) no cambia de signo en [a,b]
         4) máx {|f(a)/f'(a)|, |f(b)/f'(b)|} <= b-a
         Entonces el método converge para cualquier x0 ∈ [a,b]

        -Convergencia semilocal:
          Dado un x0 ∈ [a,b]
          * 1) 2) 3) de las condiciones de Fourier
          * 4) f(x0)f''(x0) >=0
          Entonces el método converge sólo desde ese x0

    El método converge como mínimo con orden cuadrático si la raíz es SIMPLE. Lineal en caso contrario.

    Ejemplo:

    |    f(x)= x^3 - 3*x +2
    |    x0 = -2.4
    |    xn, err = newton_raphson(f, x0, 10^-1, 10)
    """
    
    x_n = x0
    dx = f.diff(x)

    for i in range(kmax):
        x_n1 = x_n - (f(x_n) / dx(x_n))                 #Función de Newton

        if abs(x_n1 - x_n) < tol:                 
            return x_n1

        x_n = x_n1

    print("Número máximo de iteraciones superado")
    return x_n
