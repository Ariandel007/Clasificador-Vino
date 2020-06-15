from tkinter import *
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure
import red_wine_SOM as rws

win = Tk()
win.title("Clasificador de Vinos")

win.geometry('800x600')

# messagebox.showinfo('Message title','Message content')

lbl = Label(win, text="     Ingrese el factor de aprendizaje    ", font=("Arial", 12))
lbl.grid(column=0, row=0)

lbl2 = Label(win, text=" Ingrese el numero de iteraciones(se recomienda minimo 1000) ", font=("Arial", 12))
lbl2.grid(column=0, row=1)

txtFactorAprendizaje = Entry(win, width=10)

txtFactorAprendizaje.grid(column=1, row=0)

txtIteraciones = Entry(win, width=10)

txtIteraciones.grid(column=1, row=1)


def calcular():

    if float(txtFactorAprendizaje.get()) < 0.01 and float(txtIteraciones.get()) < 1:
        messagebox.showinfo('Error', 'Ingrese un numero valido')
        return


    fig = Figure(figsize=(5, 5), dpi=100)
    dimension = 11
    Rows = 40
    Cols = 40
    factorAprendisaje = float(txtFactorAprendizaje.get())
    iteraciones = int(txtIteraciones.get())
    archivo = "winequality_red.txt"


    wine_som = rws.SOM(dimension, Rows, Cols, factorAprendisaje, iteraciones, archivo)
    wine_som.algoritmo()
    a = fig.add_subplot()
    a.imshow(wine_som.retornarLabelPesos())
    canv = FigureCanvasTkAgg(fig, master=win)
    canv.draw()

    get_widz = canv.get_tk_widget()
    get_widz.grid(column=0, row=4)


btn = Button(win, text="Clasificar", command=calcular, bg="indian red", fg="white")

btn.grid(column=2, row=1)

win.mainloop()
