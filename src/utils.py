# src/utils.py

import time
import cv2
import mediapipe as mp
import math
from ultralytics import YOLO
import winsound
import threading

# =======================
# Configurações Globais
# =======================

LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
BEEP_DURATION = 1000  # Duração do beep em milissegundos

# =======================
# Variáveis de Controle
# =======================

alarm_control = False
sleep_alarm_control = False
status = ""
situation = ""
time_elapsed = 0
start_time = 0

# =======================
# Inicialização Adiada de Mediapipe e YOLO
# =======================

mp_face_mesh = None
face_mesh = None
mp_drawing = None

yolo_model = None  # Inicialização adiada do modelo YOLO


def initialize_mediapipe():
    """
    Inicializa o Mediapipe Face Mesh se ainda não estiver inicializado.
    """
    global mp_face_mesh, face_mesh, mp_drawing
    if mp_face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        mp_drawing = mp.solutions.drawing_utils


def initialize_yolo():
    """
    Inicializa o modelo YOLO se ainda não estiver inicializado.
    """
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov8l.pt")


# =======================
# Funções Utilitárias
# =======================


def initialize_video():
    """
    Inicializa a captura de vídeo.
    """
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return video_capture


def identify_phone(img):
    """
    Identifica a presença de um telefone na imagem usando o modelo YOLO.

    Args:
        img (numpy.ndarray): Imagem a ser processada.

    Returns:
        list: Coordenadas da caixa delimitadora do telefone [x1, y1, x2, y2].
    """
    initialize_yolo()
    bbox_phone = []
    result = yolo_model(img, verbose=False)
    for objects in result:
        for data in objects.boxes:
            x1, y1, x2, y2 = map(int, data.xyxy[0])
            conf = round(data.conf[0].item(), 2)
            cls = int(data.cls[0])
            if cls == 67 and conf > 0.5:
                bbox_phone = [x1, y1, x2, y2]
    return bbox_phone


def alarm():
    """
    Emite um alarme de telefone.
    """
    global alarm_control
    alarm_control = True
    winsound.Beep(2000, BEEP_DURATION)
    time.sleep(1)
    alarm_control = False


def sleep_alarm():
    """
    Emite um alarme de sono.
    """
    global sleep_alarm_control
    sleep_alarm_control = True
    winsound.Beep(2000, BEEP_DURATION)
    time.sleep(1)
    winsound.Beep(2000, BEEP_DURATION)
    time.sleep(1)
    sleep_alarm_control = False


def calculate_fps(prev_time, current_time):
    """
    Calcula os frames por segundo (FPS).

    Args:
        prev_time (float): Tempo do frame anterior.
        current_time (float): Tempo do frame atual.

    Returns:
        int: Valor de FPS.
    """
    fps = (
        1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    )
    return int(fps)


def process_frame(img, prev_frame_time):
    """
    Processa cada frame para detectar telefone e rosto.

    Args:
        img (numpy.ndarray): Frame de vídeo a ser processado.
        prev_frame_time (float): Tempo do frame anterior.

    Returns:
        tuple: Imagem processada e tempo atual.
    """
    global situation, time_elapsed, status, start_time, sleep_alarm_control

    img_resized = cv2.resize(img, (1280, 740))
    phone = identify_phone(img_resized)
    initialize_mediapipe()
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, _ = img_resized.shape

    if phone:
        draw_phone_alert(img_resized, phone)
        if not alarm_control:
            threading.Thread(target=alarm).start()
    else:
        draw_no_phone(img_resized)

    if results and results.multi_face_landmarks:
        try:
            for face in results.multi_face_landmarks:
                analyze_face(img_resized, face, w, h)
        except Exception as e:
            print(f"Error processing face: {e}")

    current_time = time.time()
    fps = calculate_fps(prev_frame_time, current_time)
    fps_text = f"FPS = {fps:.1f}"
    # Opcional: Descomente a linha abaixo para exibir FPS na imagem
    # cv2.putText(img_resized, fps_text, (105, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return img_resized, current_time


def draw_phone_alert(img, bbox_phone):
    """
    Desenha um alerta de telefone na imagem.

    Args:
        img (numpy.ndarray): Imagem a ser desenhada.
        bbox_phone (list): Coordenadas da caixa delimitadora do telefone.
    """
    cv2.rectangle(img, (100, 100), (405, 150), (0, 0, 255), -1)
    cv2.rectangle(
        img,
        (bbox_phone[0], bbox_phone[1]),
        (bbox_phone[2], bbox_phone[3]),
        (0, 0, 255),
        5,
    )
    cv2.putText(
        img,
        "ALERT! PHONE",
        (105, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
    )


def draw_no_phone(img):
    """
    Desenha um indicador de ausência de telefone na imagem.

    Args:
        img (numpy.ndarray): Imagem a ser desenhada.
    """
    cv2.rectangle(img, (100, 100), (370, 150), (0, 255, 0), -1)
    cv2.putText(
        img,
        "NO PHONE",
        (105, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
    )


def analyze_face(img, face, w, h):
    """
    Analisa o rosto para detectar sonolência.

    Args:
        img (numpy.ndarray): Imagem a ser processada.
        face (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Marcações faciais.
        w (int): Largura da imagem.
        h (int): Altura da imagem.
    """
    global situation, time_elapsed, status, start_time, sleep_alarm_control

    di1x, di1y = int(face.landmark[159].x * w), int(face.landmark[159].y * h)
    di2x, di2y = int(face.landmark[145].x * w), int(face.landmark[145].y * h)
    es1x, es1y = int(face.landmark[386].x * w), int(face.landmark[386].y * h)
    es2x, es2y = int(face.landmark[374].x * w), int(face.landmark[374].y * h)

    draw_eye_points(img, face, w, h)

    dist_di_px = math.hypot(di1x - di2x, di1y - di2y)
    dist_es_px = math.hypot(es1x - es2x, es1y - es2y)

    draw_reference_circles(img, di1x, di1y, di2x, di2y, es1x, es1y, es2x, es2y)

    determine_eye_situation(dist_es_px, dist_di_px)

    update_time_and_status()

    display_sleep_indicator(img)

    check_sleep_alarm()


def draw_eye_points(img, face, w, h):
    """
    Desenha os pontos dos olhos na imagem.

    Args:
        img (numpy.ndarray): Imagem a ser desenhada.
        face (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Marcações faciais.
        w (int): Largura da imagem.
        h (int): Altura da imagem.
    """
    for idx in LEFT_EYE:
        cx, cy = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
        cv2.circle(img, (cx, cy), 1, (0, 0, 255), 1)
    for idx in RIGHT_EYE:
        cx, cy = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
        cv2.circle(img, (cx, cy), 1, (0, 0, 255), 1)


def draw_reference_circles(
    img, di1x, di1y, di2x, di2y, es1x, es1y, es2x, es2y
):
    """
    Desenha círculos de referência na imagem.

    Args:
        img (numpy.ndarray): Imagem a ser desenhada.
        di1x (int): Coordenada x do primeiro ponto de referência.
        di1y (int): Coordenada y do primeiro ponto de referência.
        di2x (int): Coordenada x do segundo ponto de referência.
        di2y (int): Coordenada y do segundo ponto de referência.
        es1x (int): Coordenada x do terceiro ponto de referência.
        es1y (int): Coordenada y do terceiro ponto de referência.
        es2x (int): Coordenada x do quarto ponto de referência.
        es2y (int): Coordenada y do quarto ponto de referência.
    """
    points = [(di1x, di1y), (di2x, di2y), (es1x, es1y), (es2x, es2y)]
    for x, y in points:
        cv2.circle(img, (x, y), 1, (255, 0, 0), 2)


def determine_eye_situation(dist_es_px, dist_di_px):
    """
    Determina a situação dos olhos (abertos ou fechados).

    Args:
        dist_es_px (float): Distância entre os pontos es1 e es2.
        dist_di_px (float): Distância entre os pontos di1 e di2.
    """
    global situation, start_time
    if dist_es_px <= 12 and dist_di_px <= 12:
        situation = "F"  # Fechado
        start_time = time.time()
    else:
        situation = "A"  # Aberto
        start_time = time.time()


def update_time_and_status():
    """
    Atualiza o tempo e o status com base na situação dos olhos.
    """
    global situation, time_elapsed, status, start_time
    if situation == "F":
        time_elapsed = round(time.time() - start_time, 1)
    else:
        time_elapsed = int(time.time() - start_time)
    status = situation


def display_sleep_indicator(img):
    """
    Exibe o indicador de sono na imagem.

    Args:
        img (numpy.ndarray): Imagem a ser desenhada.
    """
    if not sleep_alarm_control:
        cv2.rectangle(img, (100, 30), (370, 80), (0, 255, 0), -1)
        cv2.putText(
            img,
            "AWAKE",
            (105, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
        )
    else:
        cv2.rectangle(img, (100, 30), (405, 80), (0, 0, 255), -1)
        cv2.putText(
            img,
            "ALERT! SLEEPING",
            (105, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
        )


def check_sleep_alarm():
    """
    Verifica se o alarme de sono deve ser acionado.
    """
    global time_elapsed, sleep_alarm_control
    if time_elapsed >= 2 and not sleep_alarm_control:
        threading.Thread(target=sleep_alarm).start()
