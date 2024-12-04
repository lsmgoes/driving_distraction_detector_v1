import time
import math
import winsound
from ultralytics import YOLO

# Configuração do modelo YOLO
modelo = YOLO("yolov8l.pt")
print("Modelo YOLO carregado com sucesso!")

# Configuração da duração do beep
beep_duration = 1000


# Função para identificar um celular na imagem
def identificar_celular(img):
    bboxcel = []
    resultado = modelo(img, verbose=False)
    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            x, y, w, h = dados.xyxy[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = int(dados.conf[0] * 100) / 100
            cls = int(dados.cls[0])
            if cls == 67 and conf > 0.5:  # Classe 67 = celular
                bboxcel = [x, y, w, h]
    return bboxcel


# Função para calcular a distância entre dois pontos
def calcular_distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# Função para tocar um alarme
def emitir_alarme(controle):
    controle[0] = True
    winsound.Beep(2000, beep_duration)
    time.sleep(1)
    controle[0] = False


# Função para tocar um alarme de sono
def emitir_alarme_sono(controle):
    controle[0] = True
    winsound.Beep(2000, beep_duration)
    time.sleep(1)
    winsound.Beep(2000, beep_duration)
    time.sleep(1)
    controle[0] = False
