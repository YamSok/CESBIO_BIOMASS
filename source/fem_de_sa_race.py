#! /usr/bin/env python
#

#*****************************************************************************80
#
## GLobal parameters
#
import numpy.linalg as npl
T1 = 10.0
T2 = 20.0
lambda1 = 1.0
lambda2 = 1.0

#*****************************************************************************80
#
## Compute basis functions at reference coordinates (ksi, eta)
#
def calc_phi(ksi, eta):
    phi0 = (1-ksi) * (1-eta)
    phi1 = ksi * (1-eta)
    phi2 = ksi * eta
    phi3 = (1-ksi) * eta
    return [phi0, phi1, phi2, phi3]

#*****************************************************************************80
#
## Compute gradients of basis functions at reference coordinates (ksi, eta)
#
def calc_grad_phi(ksi, eta):
    dphi0_dksi = (-1.0) * (1-eta)
    dphi0_deta = (1-ksi) * (-1.0)
    dphi1_dksi = 1.0 * (1-eta)
    dphi1_deta = ksi * (-1.0)
    dphi2_dksi = 1.0 * eta
    dphi2_deta = ksi * 1.0
    dphi3_dksi = (-1.0) * eta
    dphi3_deta = (1-ksi) * 1.0
    return np.array([[dphi0_dksi, dphi0_deta],
                     [dphi1_dksi, dphi1_deta],
                     [dphi2_dksi, dphi2_deta],
                     [dphi3_dksi, dphi3_deta]])
    

def nodeToCoord(node,grid):
    y = grid[node//len(grid)]
    x = grid[node%len(grid)]
    return x,y

    
def g_fun(node_linear_num, node_num,gamma,exact_fn):
    grid = np.linspace(0,1,node_linear_num) # nombre d'éléments
    g = np.zeros(node_num)
    T1 = np.array(Dirichlet_Nodes(node_linear_num,np.array([gamma[0],0])))
    T2 = np.array(Dirichlet_Nodes(node_linear_num,np.array([gamma[1],0])))
    T3 = np.array(Dirichlet_Nodes(node_linear_num,np.array([gamma[2],0])))
    T4 = np.array(Dirichlet_Nodes(node_linear_num,np.array([gamma[3],0])))

    for x in [T1,T2,T3,T4]:
        co = nodeToCoord(x,grid) # Pas très propre mais ça ira pour aujourd'hui
        g[x] = [exact_fn(co[0][i],co[1][i]) for i in range(len(x))]
         #g[x] = [co[0][i]**2 + co[1][i]**2 for i in range(len(x))]
    return g


def Dirichlet_Nodes(node_linear_num,gamma,output = False):
    nodes = []
    if sum(gamma == 1):
        if output:
            print("Frontier 1 is of type Dirichlet")
        nodes+=range(node_linear_num)
        if sum(gamma == 4):
            nodes.remove(0)
    if sum(gamma == 2):
        if output:
            print("Frontier 2 is of type Dirichlet")
        nodes+=list(node_linear_num*np.arange(1,node_linear_num+1)-1)
        if sum(gamma == 1):
            nodes.remove(node_linear_num-1)
    if sum(gamma == 3):
        if output:
            print("Frontier 3 is of type Dirichlet")
        nodes+=list((node_linear_num-1)*node_linear_num+np.arange(node_linear_num))
        if sum(gamma == 2):
            nodes.remove(node_linear_num**2-1)
    if sum(gamma == 4):
        if output:
            print("Frontier 4 is of type Dirichlet")
        nodes+=list(node_linear_num*np.arange(node_linear_num))
        if sum(gamma == 3):
            nodes.remove(node_linear_num*(node_linear_num-1))
    return nodes

def Dirichlet(node_linear_num,node_num,gamma,A,rhs,exact_fn,output):
    g = g_fun(node_linear_num,node_num,gamma,exact_fn)
    #plt.plot(g)
#    print("nombre de nd", len(Dirichlet_Nodes(node_linear_num,gamma,output)))
    for n in Dirichlet_Nodes(node_linear_num,gamma,output):
        A[n,:] = np.zeros(node_num)
        rhs[n] = g[n]
        A[n,n] = 1
    
    plt.plot(rhs,label="rhs")
    plt.plot(g, label="g")
    plt.legend()
    plt.show()
    #plt.imshow(A)
    #plt.show()
#*****************************************************************************80
#
## FEM solves a basic 2D linear boundary value problem in the unit square.
#
#
#    The unit square is divided into N by N squares.
#    P1 Lagrange finite element basis functions are defined.
#    The solution is sought as a piecewise linear combination of these basis functions.
#
#    This incomplete computational code is an extraction is a distributed code under the GNU LGPL license (Author: John X., USA)
#
#  Local parameters:
#
#    Local, integer ELEMENT_NUM, the number of elements.
#
#    Local, integer LINEAR_ELEMENT_NUM, the number of elements 
#    in a row in the X or Y dimension.
#
#    Local, integer NODE_LINEAR_NUM, the number of nodes in the X or Y dimension.
#
#    Local, integer NODE_NUM, the number of nodes.
#
def fem2d ( n, rhs_fn, exact_fn):

    import math
    import numpy as np
    import platform
    import scipy.linalg as la
    import matplotlib.pyplot as plt

    print ( '' )
    print ( 'FEM2D' )
    print ( '  Python version: %s' % ( platform.python_version ( ) ) )
    print ( '  Given the boundary value problem on the unit square:' )
    print ( '    - div (lambda nabla u)  = f, 0 < x < 1, 0 < y < 1' )
    print ( '  with mixed boundary conditions' )
    print ( '    bc to be determined' )
    print ( '' )
    print ( '  This program uses quadrilateral elements' )
    print ( '  and piecewise continuous bilinear basis functions.' )

    element_linear_num = n
    node_linear_num = element_linear_num + 1
    element_num = element_linear_num * element_linear_num
    node_num = node_linear_num * node_linear_num

    #
    #  MESH
    #  Set the coordinates of the nodes.
    #  The same grid is used for X and Y.
    #
    a = 0.0
    b = 1.0
    grid = np.linspace ( a, b, node_linear_num )

    if ( False ):
        print ( '' )
        print ( '  Nodes along X axis:' )
        print ( '' )
        for i in range ( 0, node_linear_num ):
            print ( '  %d  %f' %( i, grid[i] ) )
        
    #
    #  NUM. INTEGRATION
    #  Set up a quadrature rule.
    #  As a 1st step, let us define a 1d num. integration scheme, on the reference interval [0,1].
    #
    quad_num = 3

    quad_point = np.array ( ( \
        0.112701665379258311482073460022, \
        0.5, \
        0.887298334620741688517926539978 ) )

    quad_weight = np.array ( ( \
        5.0 / 18.0, \
        8.0 / 18.0, \
        5.0 / 18.0 ) )
    #####  
    #  Compute the system matrix A and right hand side RHS.
    #  There is an unknown at every node.
    #####  
    A = np.zeros ( ( node_num, node_num ) )
    rhs = np.zeros ( node_num )

    #
    #  Assembly algorithm
    #    (based on the local and global numberings)
    #

    #  We look at the element E (=square) 
    #
    #    3  *-----* 2
    #       |     |
    #       |     |
    #       | E   |
    #       |     |
    #       |     |
    #    0  *-----* 1
    #

    for ie in range(0, element_num):
        
        # Compute row and columns indices of element
        iy = ie // element_linear_num
        ix = ie - iy * element_linear_num
        
        # Compute mapping local nodes indices -> global nodes indices
        mape = np.array([iy * node_linear_num + ix, \
                        iy * node_linear_num + ix + 1, \
                        (iy + 1) * node_linear_num + ix + 1, \
                        (iy + 1) * node_linear_num + ix], dtype=int)
        
        # Compute global coordinates of local nodes (tranformation F)
        x0 = a + ix * (b - a) / element_linear_num
        y0 = a + iy * (b - a) / element_linear_num
        x1 = a + (ix+1) * (b - a) / element_linear_num
        y1 = a + iy * (b - a) / element_linear_num
        x2 = a + (ix+1) * (b - a) / element_linear_num
        y2 = a + (iy+1) * (b - a) / element_linear_num
        x3 = a + ix * (b - a) / element_linear_num
        y3 = a + (iy+1) * (b - a) / element_linear_num
        
        
        # Compute |det(DF)| and DF^-1
        absdetDF = np.abs((x1-x0)*(y2 -y0))
        invDF = np.array([[1.0 / (x1 - x0),             0.0],
                          [            0.0, 1.0 / (x1 - x0)]])

        #
        #  The 2D quadrature rule is the 'product' of X and Y copies of the 1D rule.
        #
        for l in range(0, quad_num):
        
            wl = quad_weight[l]
            s = quad_point[l]
            y = y0 + s * (y2 - y0)
            
            for k in range(0, quad_num):
                    
                wk = quad_weight[k]
                r = quad_point[k]
                x = x0 + r * (x1 - x0)
            
                #
                #  Evaluate all four basis functions, and their X and Y derivatives.
                #
                phi = calc_phi(r, s)
                grad_phi = calc_grad_phi(r, s)
                
                # Compute value of lambda
                if y0 < 0.5:
                    lambdaxy = lambda1
                else:
                    lambdaxy = lambda2
                
                # Update matrix with contributions of elemental matrix
                for i in range(0, 4):
                
                    for j in range(0, 4):
                        
                        A[mape[i], mape[j]] += absdetDF * lambdaxy * wl * wk * \
                                               np.dot(np.dot(invDF, grad_phi[j,:]), np.dot(invDF, grad_phi[i,:])) # * \
    
                # Update rights hand side
                for i in range(0, 4):
                    rhs[mape[i]] += absdetDF * wl * wk * phi[i] * rhs_fn(x, y)

    #       
    # END OF ASSEMBLY ALGORITHM
    #

    #
    #  Modify the linear system to enforce the boundary conditions where
    #  X = 0 or 1 or Y = 0 or 1.
    #

    # Update matrix with non-homogeneous Dirichlet condition on boundary y = 0
    print('rhs',rhs)
    plt.plot(rhs, label='rhs avant dirichlet')
    plt.show()

#    for ix in range(0, node_linear_num):
##        
##        A[ix, :] = 0.
##        A[ix, ix] = 1.
##        rhs[ix] = T1
##        
##    # Update matrix with non-homogeneous Dirichlet condition on boundary y = 1
##    for ix in range(0, node_linear_num):
##    
##        i = (node_linear_num - 1) * node_linear_num + ix
##    
##        A[i, :] = 0.
##        A[i, i] = 1.
##        rhs[i] = T2
        
    #
    #  Solve the linear system.
    #
    
    gamma_dirichlet = [1,2,3,4]
    Dirichlet(node_linear_num,node_num,np.array(gamma_dirichlet),A,rhs,exact_fn,False)
#    print("rhs post dirichlet",rhs)
    u = la.solve ( A, rhs )
    
    #
    #  Evaluate the exact solution at the nodes
    #   if the corresponding data are given !
    #
    uex = np.zeros ( node_linear_num * node_linear_num )
    v = 0
    for j in range ( 0, node_linear_num ):
        y = grid[j]
        for i in range ( 0, node_linear_num ):
            x = grid[i]
            if j == 0: print(j, i, uex[v])
            uex[v] = exact_fn(x, y)
            v = v + 1

    #
    #  Compute the L2 and H1 errors
    #
    
    # ETC

    #  Compare the solution and the error at the nodes.
    #
    print ( '' )
    print ( '   I     J     V    X         Y              U               Uexact' )
    print ( '' )
    v = 0
    for j in range ( 0, node_linear_num ):
        y = grid[j]
        for i in range ( 0, node_linear_num ):
            x = grid[i]
            print ( '%4d  %4d  %4d  %8f  %8f  %14g  %14g' % ( i, j, v, x, y, u[v], uex[v] ) )
            v = v + 1
        
    #
    #  Optionally, print the node coordinates.
    #
    if ( False ):
        v = 0
        for j in range ( 0, node_linear_num ):
            y = grid[j]
            for i in range ( 0, node_linear_num ):
                x = grid[i]
                print ( '%8f  %8f' % ( x, y ) )
                v = v + 1
            
    #
    #  Optionally, print the elements, listing the nodes in counterclockwise order.
    #
    if ( False ):
        e = 0
        for j in range ( 0, element_linear_num ):
            y = grid[j]
            for i in range ( 0, element_linear_num ):
                sw =   j       * node_linear_num + i
                se =   j       * node_linear_num + i + 1
                nw = ( j + 1 ) * node_linear_num + i
                ne = ( j + 1 ) * node_linear_num + i + 1
                print ( '%4d  %4d  %4d  %4d' % ( sw, se, ne, nw ) )
                e = e + 1
            
    #
    #  Optionally, plot the solution (pyplot.pcolor version)
    #
    if ( False ):
        X, Y = np.meshgrid(grid, grid)
        #Z = u.reshape((node_linear_num,node_linear_num))
        err = u - uex
        Z = err.reshape((node_linear_num,node_linear_num))
        pc = plt.pcolor(X, Y, Z)
        plt.colorbar(pc)
        plt.show()
        
    #
    #  Optionally, plot the solution and the error (pyplot.imshow version)
    #
    if ( True ):
        U = u.reshape((node_linear_num, node_linear_num))
        Uex = uex.reshape((node_linear_num, node_linear_num))
        fig,ax = plt.subplots(1,3, figsize=(15,8))
        im0 = ax[0].imshow(U)
        ax[0].set_title("Solution assemblage")
        fig.colorbar(im0,ax=ax[0])
        
        im1 = ax[1].imshow(Uex)
        ax[1].set_title("Solution exacte")
        fig.colorbar(im1,ax=ax[1])
        
        im2 = ax[2].imshow(Uex - U)
        ax[2].set_title("Uex - u")
        fig.colorbar(im2,ax=ax[2])
        plt.savefig(str(n) + ".png")
        plt.show()
        print(npl.norm(uex-u))
#        fig, (ax1, ax2) = plt.subplots(1,2)
#        p1 = ax1.imshow(u.reshape((node_linear_num, node_linear_num)))
#        plt.colorbar(p1, ax=ax1)
#        ax1.set_title("u")
#        p2 = ax2.imshow(uex.reshape((node_linear_num, node_linear_num)))
#        plt.colorbar(p2, ax=ax2)
#        ax2.set_title("uex")
#        plt.show()

    #
    #  Terminate.
    #
    print ( '' )
    print ( 'FEM2D:' )
    print ( '  Normal end of execution.' )

    return

#*****************************************************************************80
#
## EXACT_FN evaluates the exact solution.
#
    
def exact1(x,y):
    return x**2 + y**2

def exact2(x,y):
    A = 1
    k1, k2 = 1,1
    w1, w2 = np.pi, np.pi
    return A * np.sin(k1 * w1 * x) * np.cos(k2 * w2 * y)

def exact_fn ( x, y ):

  import math
  
#  C = (T2 - T1) / (0.5 / lambda1 + 0.5 / lambda2)
#  if y <= 0.5:
#    return T1 + C / lambda1 * y
#  else:
#    return T1 + C / lambda1 * 0.5 + C / lambda2 * (y - 0.5)
  return x**2 + y**2
#*****************************************************************************80
#
## RHS_FN evaluates the right hand side.
#
  
def fn1(x,y):
    return -4

def fn2(x,y):
    A = 1
    k1, k2 = 1,1
    w1, w2 = np.pi, np.pi
    Tx = A * k1 * w1 * np.cos(k1 * w1 * x) * np.cos(k2 * w2 * y)
    Txx = - A * (k1 * w1)**2 * np.sin(k1 * w1 * x) * np.cos(k2 * w2 * y)
    Ty = -A * k2 * w2 * np.sin(k1 * w1 * x) * np.sin(k2 * w2 * y)
    Tyy = - A * (k2 * w2)**2 * np.sin(k1 * w1 * x) * np.cos(k2 * w2 * y)
    return - (Txx + Tyy) 

#def rhs_fn ( x, y ):
#
#  import math
#  return -4

if ( __name__ == '__main__' ):
  
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
  
    fem2d ( 128, fn2, exact2 )

