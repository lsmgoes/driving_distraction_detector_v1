import cv2
import threading
import mediapipe as mp
from src.utils import (
    identificar_celular,
    calcular_distancia,
    emitir_alarme,
    emitir_alarme_sono,
)

# Configuração do Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Pontos de referência para os olhos
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


# Função para processar o frame e identificar o celular
def processar_frame(img, controle_alarme, controle_alarme_sono):
    celular = identificar_celular(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, _ = img.shape

    if celular:
        if not controle_alarme[0]:
            threading.Thread(
                target=emitir_alarme, args=(controle_alarme,)
            ).start()

    if results:
        try:
            for face in results.multi_face_landmarks:
                di1 = (
                    int(face.landmark[159].x * w),
                    int(face.landmark[159].y * h),
                )
                di2 = (
                    int(face.landmark[145].x * w),
                    int(face.landmark[145].y * h),
                )
                es1 = (
                    int(face.landmark[386].x * w),
                    int(face.landmark[386].y * h),
                )
                es2 = (
                    int(face.landmark[374].x * w),
                    int(face.landmark[374].y * h),
                )

                dist_di = calcular_distancia(di1, di2)
                dist_es = calcular_distancia(es1, es2)

                if dist_di <= 12 and dist_es <= 12:
                    if not controle_alarme_sono[0]:
                        threading.Thread(
                            target=emitir_alarme_sono,
                            args=(controle_alarme_sono,),
                        ).start()

        except Exception as e:
            print("Erro ao processar rosto:", e)


# Função principal
def main():
    video = cv2.VideoCapture(0)
    controle_alarme = [False]
    controle_alarme_sono = [False]

    while True:
        ret, img = video.read()
        if not ret:
            break

        img = cv2.resize(img, (1280, 740))
        processar_frame(img, controle_alarme, controle_alarme_sono)
        cv2.imshow("img", img)

        if cv2.waitKey(1) == 27:  # ESC
            break

    video.release()
    cv2
