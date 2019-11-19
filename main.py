import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

#CREAMOS PRIMERO EL DATASET CON EL QUE TE VAMOS A TRABAJAR
n = 500
p = 2
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

#PROBAMOS GRAFICAR PARA SABER SI LOS DATOS SE MUESTRAN DE MANERA CORRECTA (2 CIRCULOS)
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
#plt.show()


#UNA VEZ QUE TODO FUNCIONA CORRECTAMENTE, CREAMOS UNA CLASE CAPA NEURONAL PARA REUTILIZARLA
#POSTERIORMENTE
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f): # no conexiones, no neurona y función de activación
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1

#AHORA CREAMOS NUESTRAS FUNCIONES DE ACTIVACIÓN SIGMOIDE Y RELU
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)

#comprobamos que ambas funciones sean correctas graficandolas

#_x = np.linspace(-5, 5, 100)
#plt.plot(_x, sigm[0](_x))
#plt.show()
#plt.plot(_x, sigm[1](_x))
#plt.show()
#plt.plot(_x, relu(_x))
#plt.show()




#YA QUE TENEMOS ESTO, PODEMOS CREAR LA RED NEURONAL CON LA CLASE neural_layer

#Vamos a crear de manera recursiva una red neural de 2 a 8 capas y de vuelta a 1
#El final debe ser uno, ya que nosotros queremos que el resultado de nuestra
#red neuronal sea binario, en caso de clasificaciones mas complejas, el layer final
#puede tener n neuronas

#topology es el array que define la estructura de cuantas neuronas por layer
def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]): #lo de corchetes implica menos el último elemento
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))
    return nn

#CON ESTA FUNCIÓN Y LA CLASE neural_layer, podemos finalmente crear la red neuronal 
#definiendo un array topology



#ESTA SERÁ NUESTRA FUNCIÓN DE ENTRENAMIENTO
topology = [p, 4, 8, 1]

neural_net = create_nn(topology, sigm)
#podemos asegurarnos si todo esta bien printeando esto
#print(neural_net)
#for layer in neural_net:
    #print(layer)

#funciones de costo del error (error cuadratico medio) lambda implica una función anonima en python
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))


#ahora definimos nuestra función de entrenamiento con la red, y los datos de entrada X y salida Y
#asi como las funciones de costo, razon de aprendizaje (learning_rate o lr)
def train(neural_net, X, Y, l2_cost, lr=0.5, train=True): #podemos definirla en false si queremos ver predicciones unicamente del forward pass
    #lr muy grande va a causar que nunca converga al función y uno muy pequeño, tomará demasiado tiempo

    #primero hacemos el forward pass, para empezar a pasarlo en todas las capaz de X a Y

    #tambien creamos un vector donde podamos guardar los resultados de cada capa para poder
    #ser reutilizados por la capa anterior o posterior (dependiendo del estatus del back propagation)
    #pero le damos a este output unos datos default para iniciar
    out = [(None, X)]

    # el @ es para multiplicación matricial y lo hacemos en bucle para todas las capas
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b #esta es la suma ponderada de la capa
        #el -1 dentro de un array implica el último disponible
        a = neural_net[l].act_f[0](z) #activamos los parámetros de la capa con la función de activación sigmoide
        out.append((z, a))
    #este for implica todo el forward pass

    #si train = true, efectuamos ahora el backward pass (parte del back propagation) y el descenso del gradiente
    #para afinar la función de coste que permita minimizar el error
    if train:
        #backwardpass
        deltas = [] #ver apuntes de redes neuronales
        for l in reversed(range(0, len(neural_net))):
            z = out[l + 1][0]
            a = out[l + 1][1]
            #hay que definir en que capa estamos para saber como debemos de calcular el delta
            if l == len(neural_net) - 1: 
                #estamos en última capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a)) #esto da delta 0
            else:
                #todas las demas que no sean la última capa
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))
            #.T hace transpuesta la matriz
            #ahora implementar el descenso del gradiente para mover los parámetros al hacer back propagation
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            _W = neural_net[l].W
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
    return out[-1][1]


#y entrenamos la neural net para mandar llamar la función
#train(neural_net, X, Y, l2_cost, 0.5)


#AHORA VAMOS A ITERAR LA FUNCIÓN DE ENTRENAMIENTO PARA PROBAR QUE NUESTRA RED NEURONAL ESTE FUNCIONANDO
import time
from IPython.display import clear_output

neural_n = create_nn(topology, sigm)

loss = []

for i in range(2500):
    
    # Entrenemos a la red!
    pY = train(neural_n, X, Y, l2_cost, lr=0.05)

    if i % 500 == 0:

        print(pY)

        loss.append(l2_cost[0](pY, Y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)

        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]    

        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")

        clear_output(wait=True)
        plt.show()
        plt.plot(range(len(loss)), loss)
        plt.show()
        time.sleep(0.5)  