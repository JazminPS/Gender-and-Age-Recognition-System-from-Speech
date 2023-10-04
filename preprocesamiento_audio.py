import numpy as np
import noisereduce as nr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa
import librosa.display
from pydub.silence import split_on_silence
import soundfile as sf
import tempfile
import os
import io
import subprocess
from io import BytesIO


def reduccion_de_ruido(canal_de_audio, fs):
    # Aplicar la reducción de ruido utilizando noisereduce
    audio_filtrado_float = nr.reduce_noise(y=canal_de_audio, sr=fs)
    # Convertir el audio filtrado a formato int16
    return np.int16(audio_filtrado_float)

# Función para eliminar silencios de un audio
def eliminar_silencios(audio, sr, longitud_minima_de_silencio=100, umbral_de_silencio=-60):
    # Crear un archivo temporal para guardar el audio como WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as archivo_temporal_silencios:
        sf.write(archivo_temporal_silencios.name, audio, sr, format='wav')
        # Cargar el archivo temporal como un objeto AudioSegment
        representacion_del_audio_silencios = AudioSegment.from_wav(archivo_temporal_silencios.name)
    
    # Eliminar silencios del audio usando la función split_on_silence
    representaciones_de_audio_sin_silencios = split_on_silence(representacion_del_audio_silencios, min_silence_len=longitud_minima_de_silencio, silence_thresh=umbral_de_silencio)
    audio_sin_silencios = np.concatenate([np.array(representacion_de_audio_sin_silencios.get_array_of_samples()) / 32767 for representacion_de_audio_sin_silencios in representaciones_de_audio_sin_silencios])
    #32767 -> Este valor es debido a la conversion de audio digital en int16 a punto flotante en el rango de -1 a 1 
    # Cerrar y eliminar el archivo temporal
    archivo_temporal_silencios.close()
    os.unlink(archivo_temporal_silencios.name)
    
    return audio_sin_silencios



# Función para normalizar el audio
def normalizar_audio(audio):
    return audio / np.max(np.abs(audio))

# Función para guardar el audio en un objeto BytesIO en formato WAV
def guardar_wav_en_la_memoria(audio, sr):
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer

# Función para convertir un objeto BytesIO con un archivo WAV a MP3 utilizando FFmpeg
def convertir_wav_a_mp3(buffer_de_entrada, ruta_de_salida_normalizada):
    comando = ['ffmpeg', '-i', '-', '-codec:a', 'libmp3lame', '-qscale:a', '2', ruta_de_salida_normalizada]
    proceso = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proceso.communicate(input=buffer_de_entrada.read())
def filtro_preenfasis(senial, alpha=0.97):
    return np.append(senial[0], senial[1:] - alpha * senial[:-1])

def segmentacion_del_audio(nombre_del_archivo_de_audio_a_filtrar_Segmentar_ventanear, duracion_del_frame=20, minima_duracion_del_audio_en_seg=2, frames_totales=100):
    # Cargar el archivo de audio
    senial_original_a_filtrar_segmentar_ventanear, sr_a_filtrar_segmentar_ventanear = librosa.load(nombre_del_archivo_de_audio_a_filtrar_Segmentar_ventanear, sr=None)
    #print(sr_a_filtrar_segmentar_ventanear)
    # Verificar si el audio tiene una duración mínima de 2 segundos
    if len(senial_original_a_filtrar_segmentar_ventanear) < minima_duracion_del_audio_en_seg * sr_a_filtrar_segmentar_ventanear:
        return None
    # Aplicar el filtro de preénfasis
    senial_filtrada_con_preenfasis = filtro_preenfasis(senial_original_a_filtrar_segmentar_ventanear)
    # Calcular la longitud de la ventana senial_original_a_filtrar_segmentar_ventanear el paso (hop) en muestras
    longitud_de_frame = int(sr_a_filtrar_segmentar_ventanear * duracion_del_frame / 1000)
    #1000 -> La duracion del frame de milisegundos a segundos
    solapamiento = longitud_de_frame // 2
    # Aplicar la ventana deslizante para extraer los frames
    frames_con_preenfasis = librosa.util.frame(senial_filtrada_con_preenfasis, frame_length=longitud_de_frame, hop_length=solapamiento)
    frames_sin_preenfasis = librosa.util.frame(senial_original_a_filtrar_segmentar_ventanear, frame_length=longitud_de_frame, hop_length=solapamiento)
    # Limitar la cantidad de frames a frames_totales
    frames_con_preenfasis = frames_con_preenfasis[:, :frames_totales]
    frames_sin_preenfasis = frames_sin_preenfasis[:, :frames_totales]
    
    # Seleccionar un frame para graficar antes y después de aplicar la ventana de Hamming
    frame_indice = 0
    frame_antes_hamming = frames_con_preenfasis[:, frame_indice].copy()

    # Aplicar la ventana de Hamming a cada frame
    ventana_de_hamming = np.hamming(longitud_de_frame)
    frames_con_preenfasis = frames_con_preenfasis * ventana_de_hamming[:, np.newaxis]
    


    frame_despues_hamming = frames_con_preenfasis[:, frame_indice]

    

    return frames_con_preenfasis, frames_sin_preenfasis


def guardar_frames_en_un_archivo(frames, ruta_de_archivo_de_salida_segmentado):
    np.save(ruta_de_archivo_de_salida_segmentado, frames)

def load_frames_from_file(archivo_de_entrada):
    return np.load(archivo_de_entrada)


def procesar_audio(nombre_del_archivo_a_preprocesar):

    ##################################################################################################################
    ####################### Reducir el Ruido ##################################################################
    ##################################################################################################################


    nombre_del_archivo_de_audio_a_quitarle_el_ruido = nombre_del_archivo_a_preprocesar

    # Leer el archivo de audio
    representacion_del_audio_ruido = AudioSegment.from_mp3(nombre_del_archivo_de_audio_a_quitarle_el_ruido)
    muestras_ruido = np.array(representacion_del_audio_ruido.get_array_of_samples())
    fs_original = representacion_del_audio_ruido.frame_rate

    # Tasa de muestreo objetivo
    fs_objetivo = 32000

    # Separar los canales si el audio es estéreo
    if representacion_del_audio_ruido.channels == 2:
        muestras_del_canal_izquierdo = muestras_ruido[::2]
        muestras_del_canal_derecho = muestras_ruido[1::2]
        #print("El audio es estéreo")
    else:
        muestras_del_canal_izquierdo = muestras_ruido
        #print("El audio es monoestereo")

    # Cambiar la tasa de muestreo de cada canal a fs_objetivo
    muestras_del_canal_izquierdo = librosa.resample(muestras_del_canal_izquierdo.astype(np.float32), orig_sr=fs_original, target_sr=fs_objetivo)
    if representacion_del_audio_ruido.channels == 2:
        muestras_del_canal_derecho = librosa.resample(muestras_del_canal_derecho.astype(np.float32), orig_sr=fs_original, target_sr=fs_objetivo)

    # Aplicar la reducción de ruido a cada canal por separado
    muestras_filtradas_del_canal_izquierdo = reduccion_de_ruido(muestras_del_canal_izquierdo, fs_objetivo)
    if representacion_del_audio_ruido.channels == 2:
        muestras_filtradas_del_canal_derecho = reduccion_de_ruido(muestras_del_canal_derecho, fs_objetivo)

    # Combinar los canales filtrados (si el audio es estéreo)
    if representacion_del_audio_ruido.channels == 2:
        # Inicializa el vector muestras filtradas
        muestras_filtradas = np.empty(muestras_filtradas_del_canal_izquierdo.size + muestras_filtradas_del_canal_derecho.size, dtype=np.int16)
        muestras_filtradas[::2] = muestras_filtradas_del_canal_izquierdo
        muestras_filtradas[1::2] = muestras_filtradas_del_canal_derecho
    else:
        muestras_filtradas = muestras_filtradas_del_canal_izquierdo



    # Crear un nuevo AudioSegment con los datos filtrados
    representacion_del_audio_filtrado = AudioSegment(
        muestras_filtradas.tobytes(),
        frame_rate=fs_objetivo,
        sample_width=muestras_filtradas.dtype.itemsize,
        channels=representacion_del_audio_ruido.channels
    )

    ruta_de_archivo_de_salida_ruido = os.path.splitext(nombre_del_archivo_a_preprocesar)[0] + '_con_reduccion_de_ruido.mp3'
    # Guardar el archivo de audio filtrado en formato MP3
    representacion_del_audio_filtrado.export(ruta_de_archivo_de_salida_ruido, format="mp3")



    ##################################################################################################################
    ####################### Eliminacion de silencios ##################################################################
    ##################################################################################################################

    # Ruta al archivo de entrada
    nombre_del_archivo_de_audio_a_eliminar_silencios = ruta_de_archivo_de_salida_ruido

    # Cargar el archivo de audio
    audio_silencios, sr_silencios = librosa.load(nombre_del_archivo_de_audio_a_eliminar_silencios, sr=None)

    #print(sr_silencios)

    # Aplicar la función eliminar_silencios para obtener el audio sin silencios
    audio_sin_silencios = eliminar_silencios(audio_silencios, sr_silencios)

    # Ruta al archivo de salida
    ruta_de_salida_de_archivo_silencios = os.path.splitext(nombre_del_archivo_a_preprocesar)[0] + '_sin_silencios_largos.mp3'

    # Guardar el audio sin silencios como archivo mp3
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as archivo_temporal_silencios:
        sf.write(archivo_temporal_silencios.name, audio_sin_silencios, sr_silencios, format='wav')
        representacion_del_audio_silencios = AudioSegment.from_wav(archivo_temporal_silencios.name)
        representacion_del_audio_silencios.export(ruta_de_salida_de_archivo_silencios, format='mp3')
        # Cerrar y eliminar el archivo temporal
        archivo_temporal_silencios.close()
        os.unlink(archivo_temporal_silencios.name)


    ##################################################################################################################
    ####################################### Normalizar Audios ##################################################################
    ##################################################################################################################


    # Cargar el archivo de audio de prueba
    nombre_del_archivo_de_audio_a_normalizar = ruta_de_salida_de_archivo_silencios
    audio_a_normalizar, sr_a_normalizar = librosa.load(nombre_del_archivo_de_audio_a_normalizar, sr=None)
    #print(sr_a_normalizar)
    # Normalizar el audio
    audio_normalizado = normalizar_audio(audio_a_normalizar)

    # Guardar el audio normalizado en un objeto BytesIO en formato WAV
    buffer_wav_de_salida = guardar_wav_en_la_memoria(audio_normalizado, sr_a_normalizar)

    salida_del_audio_normalizado =  os.path.splitext(nombre_del_archivo_a_preprocesar)[0] + '_normalizado.mp3'
    convertir_wav_a_mp3(buffer_wav_de_salida, salida_del_audio_normalizado)


    ##################################################################################################################
    ####################### Preenfasis, segmentación y ventaneo ##################################################################
    ##################################################################################################################


    nombre_del_archivo_de_audio_a_filtrar_Segmentar_ventanear = salida_del_audio_normalizado
    [frames_con_preenfasis,frames_sin_preenfasis] = segmentacion_del_audio(nombre_del_archivo_de_audio_a_filtrar_Segmentar_ventanear)

    # Guardar los frames en un archivo NumPy
    ruta_de_archivo_de_salida_segmentado_con_preenfasis = os.path.splitext(nombre_del_archivo_a_preprocesar)[0] + '_segmentado_con_preenfasis.npy'
    guardar_frames_en_un_archivo(frames_con_preenfasis, ruta_de_archivo_de_salida_segmentado_con_preenfasis)


    ruta_de_archivo_de_salida_segmentado_sin_preenfasis = os.path.splitext(nombre_del_archivo_a_preprocesar)[0] + '_segmentado_sin_preenfasis.npy'
    guardar_frames_en_un_archivo(frames_sin_preenfasis, ruta_de_archivo_de_salida_segmentado_sin_preenfasis)

    return ruta_de_archivo_de_salida_ruido, ruta_de_salida_de_archivo_silencios, salida_del_audio_normalizado, ruta_de_archivo_de_salida_segmentado_con_preenfasis, ruta_de_archivo_de_salida_segmentado_sin_preenfasis