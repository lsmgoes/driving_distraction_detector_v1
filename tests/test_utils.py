import cv2
from src.utils import identificar_celular, calcular_distancia


def test_identificar_celular():
    img = cv2.imread(
        "tests/test_image.jpg"
    )  # Substitua por uma imagem de teste válida
    celular = identificar_celular(img)
    assert isinstance(celular, list)


def test_calcular_distancia():
    p1 = (0, 0)
    p2 = (3, 4)
    assert calcular_distancia(p1, p2) == 5.0
