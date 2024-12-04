from src.main import processar_frame


def test_processar_frame():
    # Mock para controle de alarmes
    controle_alarme = [False]
    controle_alarme_sono = [False]

    # Crie uma imagem mock para teste
    img = cv2.imread(
        "tests/test_image.jpg"
    )  # Substitua por uma imagem de teste válida
    processar_frame(img, controle_alarme, controle_alarme_sono)

    assert isinstance(controle_alarme[0], bool)
    assert isinstance(controle_alarme_sono[0], bool)
