import numpy as np
import matplotlib.pyplot as plt



class SOM:
    def __init__(self,dimension, Rows, Cols, factoAprendisaje, iteraciones, archivo):
        np.random.seed(1)
        self.dimension = dimension
        self.Rows = Rows;
        self.Cols = Cols
        self.rangoMax = Rows + Cols
        self.factorAprendisaje = factoAprendisaje
        self.iteraciones = iteraciones
        self.archivo = archivo
        self.dataEntrenamiento = np.loadtxt(self.archivo, delimiter=",", usecols=range(0, 11), dtype=np.float64)
        self.salida = np.loadtxt(self.archivo, delimiter=",", usecols=[11], dtype=np.int)
        self.pesos = np.random.randn(self.Rows, self.Cols, self.dimension)

    def minimoNodo(self,data, t, pesos, n_rows, m_cols):
        result = (0, 0)
        distanciaMinima = 1.0e20
        for i in range(n_rows):
            for j in range(m_cols):
                ed = self.euc_dist(pesos[i][j], data[t])
                if ed < distanciaMinima:
                    distanciaMinima = ed
                    result = (i, j)
        return result

    def euc_dist(self,v1,v2):
        return np.linalg.norm(v1 - v2)

    def manhattan_dist(self,r1,c1,r2,c2):
        return np.abs(r1-r2) * np.abs(c1-c2)

    def most_common(self, lst, n):
        if len(lst) == 0:
            return -1
        counts = np.zeros(shape=n, dtype=np.int)
        for i in range(len(lst)):
            counts[lst[i]] += 1
        return np.argmax(counts)


    def algoritmo(self):
        for s in range(self.iteraciones):
            alfa = 1.0 - (s * 1.0) / self.iteraciones
            alfaActual = alfa * self.factorAprendisaje
            rangoActual = (int)(alfa * self.rangoMax)

            t = np.random.randint(len(self.dataEntrenamiento))
            (bmu_row, bmu_col) = self.minimoNodo(self.dataEntrenamiento, t, self.pesos, self.Rows, self.Cols)

            for i in range(self.Rows):
                for j in range(self.Cols):
                    if self.manhattan_dist(bmu_row, bmu_col, i, j) < rangoActual:
                        self.pesos[i][j] = self.pesos[i][j] + alfaActual * (self.dataEntrenamiento[t] - self.pesos[i][j])

    def mostrar_pesos(self):
        print(self.pesos)

    def mostrar_una_entrada(self, T):
        t = T
        print("entrada", t)
        (bmu_row, bmu_col) = self.minimoNodo(self.dataEntrenamiento, t, self.pesos, self.Rows, self.Cols)
        print(bmu_row)
        print(bmu_col)
        print(self.pesos[bmu_row][bmu_col])

    def visualizar(self):
        print("Visualización")
        mapa = np.empty(shape=(self.Rows, self.Cols), dtype=object)
        for i in range(self.Rows):
            for j in range(self.Cols):
                mapa[i][j] = []

        for t in range(len(self.dataEntrenamiento)):
            (m_row, m_col) = self.minimoNodo(self.dataEntrenamiento, t, self.pesos, self.Rows, self.Cols)
            mapa[m_row][m_col].append(self.salida[t])

        label_pesos = np.zeros(shape=(self.Rows, self.Cols), dtype=np.int)
        for i in range(self.Rows):
            for j in range(self.Cols):
                label_pesos[i][j] = self.most_common(mapa[i][j], 20)

        plt.imshow(label_pesos)
        plt.colorbar()
        plt.show()

    def retornarLabelPesos(self):
        mapa = np.empty(shape=(self.Rows, self.Cols), dtype=object)
        for i in range(self.Rows):
            for j in range(self.Cols):
                mapa[i][j] = []

        for t in range(len(self.dataEntrenamiento)):
            (m_row, m_col) = self.minimoNodo(self.dataEntrenamiento, t, self.pesos, self.Rows, self.Cols)
            mapa[m_row][m_col].append(self.salida[t])

        label_pesos = np.zeros(shape=(self.Rows, self.Cols), dtype=np.int)
        for i in range(self.Rows):
            for j in range(self.Cols):
                label_pesos[i][j] = self.most_common(mapa[i][j], 20)

        return label_pesos