import numpy as np
import librosa
from python_speech_features import base
import os

#Función para cargar el archivo
def cargar_datos(archivo):
    return np.load(archivo)

# Funciones para LPC
def extraer_coefs_lpc(tramas, num_coefs):
    coeficientes = []
    for i in range(tramas.shape[1]):
        trama = tramas[:, i]
        coefs_lpc_trama = librosa.lpc(trama, order=num_coefs)
        coeficientes.append(coefs_lpc_trama[1:])
    return np.array(coeficientes)

# Funciones para LPCC
def lpc_a_lpcc(coefs_lpc, num_lpcc):
    coefs_lpcc = []
    for trama_lpc in coefs_lpc:
        trama_lpcc = np.zeros(num_lpcc)
        trama_lpcc[0] = trama_lpc[0]
        for m in range(1, num_lpcc):
            sumatoria = 0
            for k in range(1, m):
                sumatoria += (k / m) * trama_lpcc[k] * trama_lpc[m - k]
            trama_lpcc[m] = trama_lpc[m] + sumatoria
        coefs_lpcc.append(trama_lpcc)
    return np.array(coefs_lpcc)

# Funciones para MFCC
def extraer_caracteristicas_mfcc(tramas, tasa_muestreo, num_coefs, nfft=512, n_mels=40, fmax=None):
    longitud_trama = tramas.shape[1]
    coefs_mfcc = np.zeros((longitud_trama, num_coefs))
    for i in range(longitud_trama):
        trama = tramas[:, i]
        mfcc_trama = librosa.feature.mfcc(y=trama, sr=tasa_muestreo, n_mfcc=num_coefs, n_fft=nfft, n_mels=n_mels, fmax=fmax)
        coefs_mfcc[i, :] = mfcc_trama[:, 0]
    return coefs_mfcc

# Funciones para PLP
def hz_a_bark(frecuencias):
    return 6 * np.arcsinh(frecuencias / 600)

def bark_a_hz(barks):
    return 600 * np.sinh(barks / 6)

def filtros_triangulares(bordes_banda_critica, nfft, tasa_muestreo):
    num_filtros = len(bordes_banda_critica) - 2
    indices_bines = np.floor((nfft + 1) * bordes_banda_critica / tasa_muestreo)
    bancos_filtros = np.zeros((num_filtros, nfft // 2 + 1))

    for i in range(num_filtros):
        for j in range(int(indices_bines[i]), int(indices_bines[i+1])):
            bancos_filtros[i, j] = (j - indices_bines[i]) / (indices_bines[i+1] - indices_bines[i])
        for j in range(int(indices_bines[i+1]), int(indices_bines[i+2])):
            bancos_filtros[i, j] = (indices_bines[i+2] - j) / (indices_bines[i+2] - indices_bines[i+1])
    return bancos_filtros

def extraer_coefs_plp_de_tramas(tramas, tasa_muestreo, paso_ventana=0.01, numcep=13, nfilt=26, nfft=512):
    longitud_trama = tramas.shape[1]
    coefs_plp = np.zeros((longitud_trama, numcep))

    for i in range(longitud_trama):
        trama = tramas[:, i]
        espectro_potencia_trama = np.abs(np.fft.rfft(trama, n=nfft)) ** 2
        bordes_banda_critica = bark_a_hz(np.linspace(hz_a_bark(0), hz_a_bark(tasa_muestreo // 2), nfilt + 2))
        bancos_filtros = filtros_triangulares(bordes_banda_critica, nfft, tasa_muestreo)

        # Aplicar el banco de filtros a la señal
        energia_bancos_filtros = np.dot(espectro_potencia_trama, bancos_filtros.T)
        energia_bancos_filtros = np.where(energia_bancos_filtros == 0, np.finfo(float).eps, energia_bancos_filtros)

        # Compresión de la señal
        energia_bancos_filtros = np.log(energia_bancos_filtros)

        # Obtener los coeficientes PLP
        coefs_plp[i, :] = base.dct(energia_bancos_filtros)[:numcep]

    return coefs_plp

# Función para guardar
def guardar_coefs(archivo, coeficientes):
    np.save(archivo, coeficientes)

# Extraer coeficientes
def extraer_caracterisiticas(archivo_de_tramas):
    archivo_datos = archivo_de_tramas
    datos = cargar_datos(archivo_datos)
    tasa_muestreo = 32000
    ruta_archivo = os.path.splitext(archivo_de_tramas)[0]

    # Extraer y guardar coefs LPC
    coefs_lpc = extraer_coefs_lpc(datos, 13)
    archivo_lpc = ruta_archivo +'_lpc.npy'
    guardar_coefs(archivo_lpc, coefs_lpc)

    # Extraer y guardar coefs LPCC
    coefs_lpcc = lpc_a_lpcc(extraer_coefs_lpc(datos, 14), 13)
    archivo_lpcc = ruta_archivo +'_lpcc.npy'
    guardar_coefs(archivo_lpcc, coefs_lpcc)

    # Extraer y guardar coefs MFCC
    coefs_mfcc = extraer_caracteristicas_mfcc(datos, tasa_muestreo, 13)
    archivo_mfcc = ruta_archivo +'_mfcc.npy'
    guardar_coefs(archivo_mfcc, coefs_mfcc)

    # Extraer y guardar coefs PLP
    coefs_plp = extraer_coefs_plp_de_tramas(datos, tasa_muestreo)
    archivo_plp = ruta_archivo +'_plp.npy'
    guardar_coefs(archivo_plp, coefs_plp)

    return archivo_lpc, archivo_lpcc, archivo_mfcc, archivo_plp

