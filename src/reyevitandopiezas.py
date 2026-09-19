#Importacion de librerias
import urllib.request #Para descargar recursos remotos
import importlib.util #Carga e importacion de modulos
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#Fork del repositorio original "baile"
url = "https://raw.githubusercontent.com/GerryPad/IA/refs/heads/master/src/SimpleSearch.py"

urllib.request.urlretrieve(url, "SimpleSearch.py")

#Importamos los algoritmos de busqueda
import SimpleSearch as sp

import math
import time #Para tomar el tiempo de cada algoritmo
from IPython.display import clear_output #Para las animaciones del tablero
tam_tab = 16

def esta_amenazada(cx, cy, ocupadas):
  direcciones_ortogonales = [(1, 0), (-1, 0), (0, 1), (0, -1)] #Movimientos de torre
  direcciones_diagonales = [(1, 1), (1, -1), (-1, 1), (-1, -1)] #Movientos de alfil
  saltos_caballo = [(2, 1), (2, -1), (-2, 1), (-2, -1),(1, 2), (1, -2), (-1, 2), (-1, -2)]

  #Comprobar si la casilla coincide con el salto de algun caballo
  for dx, dy in saltos_caballo:
      if (cx + dx, cy + dy) in caballos:
          return True

  #Proyectamos un rayo en cada horizontal y vertical hasta el limite del tablero
  for dx, dy in direcciones_ortogonales:
      act_x, act_y = cx + dx, cy + dy
      while 0 <= act_x < tam_tab and 0 <= act_y < tam_tab:
          #Si el primer obstaculo es una torre, la casilla esta amenazada
          if (act_x, act_y) in ocupadas:
              if (act_x, act_y) in torres:
                  return True
              break #Cualquier otra pieza en esta direccion bloquea ataques de torre
          act_x += dx
          act_y += dy

  #Proyectamos rayo en diagonales hasta el limite del tablero...
  for dx, dy in direcciones_diagonales:
      act_x, act_y = cx + dx, cy + dy
      while 0 <= act_x < tam_tab and 0 <= act_y < tam_tab:
        #Si el primer obstaculo es una alfil, la casilla esta amenazada
          if (act_x, act_y) in ocupadas:
              if (act_x, act_y) in alfiles:
                  return True
              break #Cualquier otra pieza en esta direccion bloquea ataques de alfil
          act_x += dx
          act_y += dy

  return False #Si no hay amenazas

#Funcion para filtrar estados sucesores; no estan permitidas las capturas del rey ni las casillas en jaque.
def descartar_amenazas(sucesores):
    ocupadas = set(alfiles + torres + caballos)

    sucesores_seguros = []

    for nodo in sucesores:
      #Si la casilla propuesta esta ocupada, no se considera para los sucesores
      if nodo in ocupadas:
        continue
      x = nodo[0]
      y = nodo[1]
      #Si la casilla se encuentra libre de amenaza, se añade al estado a los sucesores
      if esta_amenazada(x, y, ocupadas) == False:
          sucesores_seguros.append(nodo)

    return sucesores_seguros

#Funcion de transicion requerida por los algoritmos de busqueda, recibe el estado actual del rey
def sucesor(nodo):
    x, y = nodo.state
    #Generamos los 8 posibles sucesores que no salgan del tablero, que no sean la casilla actual del rey
    sucesores_con_amenazas = [
        (x + dx, y + dy)
        for dx in [-1, 0, 1]
          if not (x+dx > 15 or x+dx < 0)
        for dy in [-1, 0, 1]
          if not (y+dy > 15 or y+dy < 0)
        if not (dx == 0 and dy == 0)
    ]

    #Filtramos para quedarnos solo con las casillas libres
    sucesores_sin_amenaza = descartar_amenazas(sucesores_con_amenazas)

    #Cada casilla legal se encapsula en un nuevo nodo, cuyo padre es la casilla actual y aumentamos la profundidad en 1
    hijos = []
    for coord in sucesores_sin_amenaza:
        hijo = sp.node(coord, parent=nodo, depth=nodo.depth + 1)
        hijos.append(hijo)
    return hijos

#Funcion para visualizar la configuracion del tablero: posiciones del rey y piezas rivales
def plot_tablero(state, alfiles, torres, caballos, final, inicio):
    n=tam_tab
    tablero=[[(i+j)%2 for j in range(n)] for i in range(n)]
    fig, ax=plt.subplots(figsize=(8,8))
    ax.imshow(tablero, cmap="YlGn")

    if state is inicio:
      #Para cuando el rey aun no se ha movido
      ax.text(state[0],state[1],"♔", ha="center", va="center", fontsize=30, color="magenta")
    else:
      #Para marcar la casilla de origen
      ax.text(inicio[0],inicio[1],"GO!", ha="center", va="center", fontsize=15, color="red")

    if state is final:
      #El rey llego a la meta
      ax.text(state[0],state[1],"♔", ha="center", va="center", fontsize=30, color="magenta")
    else:
      #El rey va de camino a la meta
      ax.text(state[0],state[1],"♔", ha="center", va="center", fontsize=30, color="magenta")
      ax.text(final[0],final[1],"✔", ha="center", va="center", fontsize=30, color="red")

    #Dibujo de las posiciones de las piezas rivales
    for r,c in alfiles:
      ax.text(r,c,"♝", ha="center", va="center", fontsize=30, color="cyan")
    for r,c in torres:
      ax.text(r,c,"♜", ha="center", va="center", fontsize=30, color="cyan")
    for r,c in caballos:
      ax.text(r,c,"♞", ha="center", va="center", fontsize=30, color="cyan")
    plt.show()

#Calcula la distancia Chebyshev entre la posicion actual del rey y la meta
def chebyshev(*nodos):
    nodo = nodos[0]
    x = nodo.state[0]
    y = nodo.state[1]
    #Dado los movimentos del rey, este puede llegar en no menos que la mayor de sus diferencias de coordenadas
    maximo = max(abs(goal[0]-x), abs(goal[1]-y))
    return maximo

#Variacion de la distancia Manhattan
def manhatthan(*nodos):
    nodo = nodos[0]
    x = nodo.state[0]
    y = nodo.state[1]
    #Se le hace el ajuste de dividir por 2 y tomar el techo para no sobreestimar el costo real
    total = math.ceil((abs(goal[0]-x) +abs(goal[1]-y) ) / 2)
    return total

#Funcion para determinar si el estado actual es el estado meta y hemos hallado solucion
def meta(*nodos):
    nodo = nodos[0]
    posicion_actual = nodo.state
    return posicion_actual == goal

#La función animar_camino como su nombre lo indica nos sirve para animar la ruta que toma el rey para llegar a la meta.
#Hacemos uso de la función plot_tablero para obtener una imagen por cada movimiento que hace el rey y simular una animación.
def animar_camino(resultado_busqueda, alfiles, torres, caballos):
    camino = resultado_busqueda.getPath()
    meta_alcanzada = camino[-1][0]
    inicio = camino[0][0]

    for nodo in camino:
        posicion_rey = nodo[0]
        clear_output(wait=True)
        plot_tablero(posicion_rey, alfiles, torres, caballos, meta_alcanzada, inicio)
        time.sleep(0.5)

#Estructuras para almacenamiento de soluciones
datos_tabla = []
soluciones_BFS=[]
soluciones_DFS=[]
soluciones_A_SH=[]
soluciones_A_M=[]
soluciones_A_C=[]

#Configuracion de 3 escenarios (instancias)
casos = [
    {
        "nombre": "Fácil",
        "start_state": (8, 4),
        "alfiles": [(9, 4)],
        "torres": [(12, 5)],
        "caballos": [(8, 8), (12, 3)],
        "goal": (1, 0)
    },
    {
        "nombre": "Medio",
        "start_state": (6, 0),
        "alfiles": [(13, 9), (7, 5), (4, 10), (6, 4)],
        "torres": [(13, 3), (3, 12), (3,8), (5, 15)],
        "caballos": [(8, 8), (12, 3), (0, 0), (9, 12)],
        "goal": (15, 14)
    },
    {
        "nombre": "Difícil",
        "start_state": (1, 4),
        "alfiles": [(13, 9), (7, 5), (4, 10), (5,1)],
        "torres": [(13, 3), (0, 14), (15,1), (10,15)],
        "caballos": [(8, 8), (11, 3), (0, 0), (10, 12)],
        "goal": (12, 2)
    }
]

for c in casos:
    #Inicializamos el nodo raiz y extraemos los datos de la instancia actual
    start = sp.node(c['start_state'], depth=0, parent=None)
    alfiles = c['alfiles']
    torres = c['torres']
    caballos = c['caballos']
    goal = c['goal']

    #Verificar que la meta no este en jaque
    if esta_amenazada(goal[0], goal[1], set(alfiles + torres + caballos)):
        print("El rey no puede llegar a la meta porque está en jaque")
        continue

    #Busqueda en anchura (BFS)
    bfs = sp.TreeSearch(start, sucesor, meta, strategy="bfs")
    tb = time.time()
    r1 = bfs.find(max_iter=256)
    tf = time.time()
    tiempo_bfs = tf - tb
    if r1 is None:
      print("No se encontró una solución. Instancia:", c['nombre'])
    else:
      soluciones_BFS.append(r1)

    #Busqueda en profundidad (DFS)
    dfs = sp.TreeSearch(start, sucesor, meta, strategy="dfs")
    td = time.time()
    r2 = dfs.find(max_iter=256)
    tf = time.time()
    tiempo_dfs = tf - td
    if r2 is None:
      print("No se encontró una solución. Instancia:", c['nombre'])
    else:
      soluciones_DFS.append(r2)

    #Busqueda de costo uniforme (A* sin heuristica)
    bas=sp.TreeSearch(start, sucesor, meta, strategy="a*")
    tbas = time.time()
    r3 = bas.find(max_iter=256)
    tfbas = time.time()
    tiempo_bas = tfbas - tbas
    if r3 is None:
      print("No se encontró una solución. Instancia:", c['nombre'])
    else:
      soluciones_A_SH.append(r3)

    #Busqueda con heuritica Manhattan
    bas2 = sp.TreeSearch(start, sucesor, meta, strategy="a*", heuristic=manhatthan)
    tbas2 = time.time()
    r4 = bas2.find(max_iter=256)
    tfbas2 = time.time()
    tiempo_bas2 = tfbas2 - tbas2
    if r4 is None:
      print("No se encontró una solución. Instancia:", c['nombre'])
    else:
      soluciones_A_M.append(r4)

    #Busqueda con heuritica Chebyshev
    bas3=sp.TreeSearch(start, sucesor, meta, strategy="a*", heuristic=chebyshev)
    tbas3 = time.time()
    r5 = bas3.find(max_iter=256)
    tfbas3 = time.time()
    tiempo_bas3 = tfbas3 - tbas3
    if r5 is None:
      print("No se encontró una solución. Instancia:", c['nombre'])
    else:
      soluciones_A_C.append(r5)

    #Banderas para identificar instancias inconclusas
    easy = False
    medium = False
    hard = False
    if r1 is None or r2 is None or r3 is None or r4 is None or r5 is None:
      if c['nombre'] == "Fácil":
        easy = True
      elif c['nombre'] == "Medio":
        medium = True
      elif c['nombre'] == "Difícil":
        hard = True
      continue

    #Extraemos los nodos expandidos, tiempo de ejecucion, longitud y costo de cada solucion
    datos_tabla.append({"Instancia": c['nombre'], "Estrategia": "BFS", "Nodos expandidos": bfs.iterations, "Tiempo (s)": tiempo_bfs, "Longitud": len(r1.getPath())-1, "Costo": r1.cost})
    datos_tabla.append({"Instancia": c['nombre'], "Estrategia": "DFS", "Nodos expandidos": dfs.iterations, "Tiempo (s)": tiempo_dfs, "Longitud": len(r2.getPath())-1, "Costo": r2.cost})
    datos_tabla.append({"Instancia": c['nombre'], "Estrategia": "A* (Sin heurística)", "Nodos expandidos": bas.iterations, "Tiempo (s)": tiempo_bas, "Longitud": len(r3.getPath())-1, "Costo": r3.cost})
    datos_tabla.append({"Instancia": c['nombre'], "Estrategia": "A* (Manhatthan)", "Nodos expandidos": bas2.iterations, "Tiempo (s)": tiempo_bas2, "Longitud": len(r4.getPath())-1, "Costo": r4.cost})
    datos_tabla.append({"Instancia": c['nombre'], "Estrategia": "A* (Chevishev)", "Nodos expandidos": bas3.iterations, "Tiempo (s)": tiempo_bas3, "Longitud": len(r5.getPath())-1, "Costo": r5.cost})

tabla_comparativa = pd.DataFrame(datos_tabla)
tabla_comparativa

#Esta sección de código nos ayuda a animar algún camino recorrido por el rey para llegar a la meta en alguna de las instancias y para algún algoritmo de búsqueda.
#Use soluciones_BFS=[] para animar caminos BFS
#Use soluciones_DFS=[] para animar caminos DFS
#Use soluciones_A_SH=[] para animar caminos A* sin heurística
#Use soluciones_A_M=[] para animar caminos A* con heurística Manhatthan modificada
#Use soluciones_A_C=[] para animar caminos A* con heurística Chevishev
#Use un valor de op=0 para animar la instancia facil, un valor de op=1 para la media y un op=2 para la dificil
op = 0
if easy == True:
  if op != 0:
    animar_camino(soluciones_A_C[op], casos[op]["alfiles"], casos[op]["torres"], casos[op]["caballos"])
  else:
    print("No se encontró solución")
elif medium == True:
  if op != 1:
    animar_camino(soluciones_A_C[op], casos[op]["alfiles"], casos[op]["torres"], casos[op]["caballos"])
  else:
    print("No se encontró solución")
elif hard == True:
  if op != 2:
    animar_camino(soluciones_A_C[op], casos[op]["alfiles"], casos[op]["torres"], casos[op]["caballos"])
  else:
    print("No se encontró solución")
else:
  animar_camino(soluciones_A_C[op], casos[op]["alfiles"], casos[op]["torres"], casos[op]["caballos"])
