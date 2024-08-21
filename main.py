from fastapi import FastAPI
from datetime import datetime
import pandas as pd

app = FastAPI()

# Cargar los datos de películas (simularemos la carga desde el CSV)
data = pd.read_csv(r"./Data/Datos_de_Pelicula.csv")
cast_data = pd.read_csv(r"./Data/cast_limpio.csv")
crew_data = pd.read_csv(r"./Data/crew_limpio.csv")


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Películas"}

# Función para consultar la cantidad de películas por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    # Convertir la columna release_date a datetime
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    
    # Traducir mes a número
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, 
        "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, 
        "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_num = meses.get(mes.lower())
    
    if mes_num is None:
        return {"error": "Mes no válido. Ingrese un mes en español."}
    
    # Filtrar por el mes especificado
    cantidad = data[data['release_date'].dt.month == mes_num].shape[0]
    
    return {"message": f"{cantidad} películas fueron estrenadas en el mes de {mes}"}

# Función para consultar la cantidad de películas por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    # Convertir la columna release_date a datetime
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    
    # Traducir día a número
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, 
        "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia_num = dias.get(dia.lower())
    
    if dia_num is None:
        return {"error": "Día no válido. Ingrese un día en español."}
    
    # Filtrar por el día especificado
    cantidad = data[data['release_date'].dt.weekday == dia_num].shape[0]
    
    return {"message": f"{cantidad} películas fueron estrenadas en los días {dia}"}

# Función para consultar título y su score
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    # Buscar la película por título
    film = data[data['title'].str.lower() == titulo.lower()]
    
    if film.empty:
        return {"error": "Título no encontrado."}
    
    # Extraer los datos relevantes
    titulo_film = film.iloc[0]['title']
    release_date = film.iloc[0]['release_date']
    score = film.iloc[0]['vote_average']
    
    return {
        "titulo": titulo_film,
        "año_estreno": release_date.year,
        "score": score
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    # Buscar la película por título
    film = data[data['title'].str.lower() == titulo.lower()]
    
    if film.empty:
        return {"error": "Título no encontrado."}
    
    # Extraer el número de votos y el promedio
    votos = film.iloc[0]['vote_count']
    promedio_votos = film.iloc[0]['vote_average']
    
    # Verificar si tiene al menos 2000 votos
    if votos < 2000:
        return {"message": f"La película '{titulo}' no tiene suficientes valoraciones (mínimo 2000)."}
    
    # Si cumple con el mínimo de votos
    titulo_film = film.iloc[0]['title']
    
    return {
        "titulo": titulo_film,
        "cantidad_votos": votos,
        "promedio_votos": promedio_votos
    }
