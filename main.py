# !pip install opencv-python mediapipe ultralytics
# Necessário importar o CUDA
import time
import cv2
import mediapipe as mp  # detecção facial
import math
from ultralytics import YOLO  # detecção de objetos em tempo real, nesse caso o smartphone
import winsound  # utilizada para emitir beeps sonoros
import threading

# Configuração do mediapipe
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Pesos pré-treinados do modelo YOLO
modelo = YOLO("yolov8l.pt") # n=precisão baixa / m=precisão média / x=alta precisão;

# Configuração da captura de vídeo (inicia a webcam)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# video = cv2.VideoCapture("IMG_0439.MOV")
# video = cv2.VideoCapture("SONO.mp4")

# Variáveis de controle
situacao = ""
tempo = 0
status = ""
inicio = ""

# Pontos de referência para os olhos.   Referência (direito)  (esquerdo) de quem está olhando p/ monitor
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Configuração da duração do beep
beep = 1000  # Esse valor define a velocidade do beep

# Função para identificar um celular na imagem
def identificarCelular(img):
    bboxcel = []
    resultado = modelo(img, verbose=False)
    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            x, y, w, h = dados.xyxy[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = int(dados.conf[0] * 100) / 100
            cls = int(dados.cls[0])
            # Seleciona a classe/objeto que deseja identificar. Ex: 0=pessoas; 2=carros; 67=cellphone
            if cls == 67 and conf > 0.5: # limiar
                bboxcel = [x, y, w, h]
    return bboxcel


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

controleAlarme = False
controleAlarmeSono = False

def alarme():
    global controleAlarme
    controleAlarme = True
    winsound.Beep(2000, beep)  # Emite um beep sonoro
    time.sleep(1)
    controleAlarme = False


def alarmeSono():
    global controleAlarmeSono
    controleAlarmeSono = True
    winsound.Beep(2000, beep)  # Emite um beep sonoro
    time.sleep(1)
    winsound.Beep(2000, beep)  # Emite um beep sonoro
    time.sleep(1)
    controleAlarmeSono = False

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# videoFinal = cv2.VideoWriter(f'final.avi', fourcc, 20.0, (1280, 740))

while True:
    _,img = video.read()
    img = cv2.resize(img, (1280, 740))  # Redimensiona a imagem capturada

    # Identificação do celular na imagem
    celular = identificarCelular(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem para o espaço de cores RGB
    results = faceMesh.process(imgRGB)  # Processa a imagem para detecção facial
    h, w, _ = img.shape  # Obtém a altura e largura da imagem

    # Verificação se um celular foi detectado
    if celular:
        cv2.rectangle(img, (100, 100), (405, 150), (0, 0, 255), -1)  # Desenha um retângulo vermelho
        cv2.rectangle(img, (celular[0], celular[1]), (celular[2], celular[3]), (0, 0, 255), 5)  # Desenha vermelho ao redor do celular
        cv2.putText(img, 'ALERTA! CELULAR', (105, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  # Adiciona texto indicando alerta de celular
        if not controleAlarme:
            threading.Thread(target=alarme).start()

    else:
        cv2.rectangle(img, (100, 100), (370, 150), (0, 255, 0), -1)  # Desenha um retângulo verde (370 larg / 150 alt)
        cv2.putText(img, 'SEM CELULAR', (105, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  # Adiciona texto indicando a ausência de celular

    # Verificação se há resultados da detecção facial
    if results:
        try:
            for face in results.multi_face_landmarks:
                # Pontos de referência dos olhos
                di1x, di1y = int((face.landmark[159].x) * w), int((face.landmark[159].y) * h)
                di2x, di2y = int((face.landmark[145].x) * w), int((face.landmark[145].y) * h)
                es1x, es1y = int((face.landmark[386].x) * w), int((face.landmark[386].y) * h)
                es2x, es2y = int((face.landmark[374].x) * w), int((face.landmark[374].y) * h)

                # Desenha círculos nos pontos de referência dos olhos
                for x in LEFT_EYE:
                    cx, cy = int((face.landmark[x].x) * w), int((face.landmark[x].y) * h)
                    cv2.circle(img, (cx, cy), 1, (0, 0, 255), 1)

                for x in RIGHT_EYE:
                    cx, cy = int((face.landmark[x].x) * w), int((face.landmark[x].y) * h)
                    cv2.circle(img, (cx, cy), 1, (0, 0, 255), 1)

                # Cálculo da distância entre os pontos de referência dos olhos
                distDiPx = math.hypot(di1x - di2x, di1y - di2y)
                distEsPx = math.hypot(es1x - es2x, es1y - es2y)

                # Desenha círculos nos pontos de referência dos olhos
                cv2.circle(img, (di1x, di1y), 1, (255, 0, 0), 2)
                cv2.circle(img, (di2x, di2y), 1, (255, 0, 0), 2)
                cv2.circle(img, (es1x, es1y), 1, (255, 0, 0), 2)
                cv2.circle(img, (es2x, es2y), 1, (255, 0, 0), 2)

                # Verificação da situação dos olhos (abertos ou fechados)
                print(distEsPx,distDiPx)
                if distEsPx <= 12 and distDiPx <= 12:
                    situacao = 'F'  # Olhos fechados
                    if status != situacao:
                        inicio = time.time()  # Marca o início do fechamento dos olhos
                else:
                    situacao = 'A'  # Olhos abertos
                    inicio = time.time()  # Marca o início da abertura dos olhos
                    tempo = int(time.time() - inicio)  # Calcula o tempo que os olhos estão abertos

                if situacao == 'F':
                    tempo = round(time.time() - inicio, 1)  # Calcula o tempo que os olhos estão fechados

                status = situacao

                # Adiciona texto indicando a situação dos olhos

                if not controleAlarmeSono:
                    cv2.rectangle(img, (100, 30), (370, 80), (0, 255, 0),-1)  # Desenha um retângulo verde (370 larg / 150 alt)
                    cv2.putText(img, 'ACORDADO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                else:
                    cv2.rectangle(img, (100, 30), (405, 80), (0, 0, 255), -1)
                    cv2.putText(img, 'ALERTA! DORMINDO', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                # Verificação de tempo com os olhos fechados
                if tempo >= 2: # 2 segundos
                    if not controleAlarmeSono:
                        threading.Thread(target=alarmeSono).start()

        except:
            pass

    # Calculate the FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = int(fps)
    # converting the fps to string so that we can display it on frame
    # by using putText function
    # fps = str(fps)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    # cv2.putText(img, fps_text, (105, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # videoFinal.write(img)

    cv2.imshow('img', img) # Exibe a imagem na janela

    if cv2.waitKey(1)==27: # Aguarda 1 milissegundo
        break

