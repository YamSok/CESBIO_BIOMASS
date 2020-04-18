def bestSigma(nitermax):
    fig,ax = plt.subplots(1,2, figsize=(18,5))
    nit = [] # Tableau du nombre d'itérations par valeur du paramètre sigma
    testData = [10.**k for k in np.arange(-2,2,1)]
    for sigma in testData:
        (alpha,xtilde,CF,err)=Nesterov(A,y,sigma,nitermax,1e-2)
        nit.append(len(err))

        im1 = ax[0].plot(CF[2:400],  label='sigma = ' + str(sigma))
        ax[0].set_title("Fonction de coût en fonction des itérations")
        ax[0].set_xlabel("itération")
        ax[1].set_ylabel("Fonction de coût")
        ax[0].legend()

        im2 = ax[1].plot(err[2:],  label='sigma = ' + str(sigma))
        ax[1].set_title("Erreur absolue en fonction des itérations")
        ax[1].set_xlabel("itération")
        ax[1].set_ylabel("|| xn+1 - xn ||")

        ax[1].legend()

    plt.show()

    plt.plot(testData,nit)
    plt.scatter(testData,nit)
    plt.xscale("log")
    plt.xlabel("sigma (log)")
    plt.ylabel("Itérations")
    plt.title("Nombre d'itération en fonction de sigma")
    plt.show()

def recoNester(p,x):
    A=np.random.randn(p,n) #La matrice de mesure
    y=A.dot(x)        #Les mesures
    (alpha,xtilde,CF,err_loc) = Nesterov(A,y,0.1,2000,1e-5)
    fig,ax = plt.subplots(1,2,figsize=(10,3))

    ax[0].plot(xtilde)
    ax[0].set_title("Signal reconstruit avec " + str(p) + " mesures de x")

    ax[1].plot(np.abs(x-xtilde))
    ax[1].set_title("|x - xtilde|")

    plt.tight_layout()
    plt.show()

testData = [8,25,50,100]
for p in testData:
    recoNester(p,x)


# Fonction testant l'influence du nombre de mesure sur l'erreur relative || (x - xtilde) / x ||
# Plot de l'erreur relative en fonction des valeurs p de test
def testReco(testData,n,A,sigma,nitermax,seuil,x,Neste = True):
    err = []
    for p in testData:
        A=np.random.randn(p,n) #La matrice de mesure
        y=A.dot(x)        #Les mesures
        (alpha,xtilde,CF,err_loc) = Nesterov(A,y,sigma,nitermax,seuil)
        err.append(npl.norm((x-xtilde)/x))
    plt.plot(testData,err)
    plt.scatter(testData,err)
    plt.xlabel("Nombre de mesures (p)")
    plt.ylabel("Erreur relative")
    plt.title("Erreur relative en fonction du nombre de mesures pour reconstruire le signal")
    plt.show()
    return err
testData = [1,4,8,10,20,40,80,100,200,1000] # Valeurs de nombre de mesures (p) à tester
diff = testReco(testData,n,A,0.1,3000,1e-5,x)
"""
Ce graphe ne nous donne qu'une indication concernant l'influence du nombre de mesure car il y a de l'aléa dans chaque test introduit par la matrice A.
On peut cependant discener une forte décroissance de l'erreur relative entre 4 et environ 25 mesures. A partir de 50 mesures, le gain en erreur n'est plus aussi significatif.
"""
