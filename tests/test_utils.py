# tests/test_utils.py

import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

# Importar as funções e variáveis que serão testadas
from src.utils import (
    initialize_video,
    identify_phone,
    calculate_fps,
    determine_eye_situation,
    process_frame,
    draw_phone_alert,
    draw_no_phone,
    analyze_face,
    alarm,
    sleep_alarm,
)


class TestUtils(unittest.TestCase):
    """Classe de testes unitários para as funções em utils.py"""

    @patch("src.utils.cv2.VideoCapture")
    def test_initialize_video(self, mock_video_capture):
        """Testa se a função initialize_video inicializa corretamente a captura de vídeo."""
        mock_instance = MagicMock()
        mock_video_capture.return_value = mock_instance

        video = initialize_video()
        mock_video_capture.assert_called_with(0, cv2.CAP_DSHOW)
        self.assertEqual(video, mock_instance)

    @patch("src.utils.initialize_yolo")
    @patch("src.utils.YOLO")
    def test_identify_phone(self, mock_yolo_class, mock_initialize_yolo):
        """Testa a função identify_phone para identificar a presença de um telefone na imagem."""
        # Configurar o mock do modelo YOLO
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Criar uma imagem dummy
        img = np.zeros((640, 480, 3), dtype=np.uint8)

        # Configurar o resultado do modelo YOLO
        mock_box = MagicMock()
        mock_box.xyxy = [[100, 150, 200, 250]]
        mock_box.conf = [0.6]
        mock_box.cls = [67]  # Supondo que a classe 67 corresponda a 'phone'
        mock_result = [MagicMock(boxes=[mock_box])]
        mock_model.return_value = mock_result

        bbox_phone = identify_phone(img)
        self.assertEqual(bbox_phone, [100, 150, 200, 250])
        mock_initialize_yolo.assert_called_once()
        mock_yolo_class.assert_called_once_with("yolov8l.pt")

    def test_calculate_fps(self):
        """Testa a função calculate_fps para calcular corretamente os FPS."""
        prev_time = 1000
        current_time = 1001
        fps = calculate_fps(prev_time, current_time)
        self.assertEqual(fps, 1)

        # Testar divisão por zero
        fps_zero = calculate_fps(1000, 1000)
        self.assertEqual(fps_zero, 0)

    @patch("src.utils.time")
    def test_determine_eye_situation_closed(self, mock_time):
        """Testa determine_eye_situation quando os olhos estão fechados."""
        global situation, start_time
        mock_time.time.return_value = 1000
        determine_eye_situation(10, 10)  # dist_es_px <= 12 e dist_di_px <= 12
        self.assertEqual(situation, "F")
        self.assertEqual(start_time, 1000)

    @patch("src.utils.time")
    def test_determine_eye_situation_open(self, mock_time):
        """Testa determine_eye_situation quando os olhos estão abertos."""
        global situation, start_time
        mock_time.time.return_value = 1000
        determine_eye_situation(15, 15)  # dist_es_px > 12 ou dist_di_px > 12
        self.assertEqual(situation, "A")
        self.assertEqual(start_time, 1000)

    @patch("src.utils.threading.Thread")
    @patch("src.utils.analyze_face")
    @patch("src.utils.draw_no_phone")
    @patch("src.utils.draw_phone_alert")
    @patch("src.utils.face_mesh.process")
    @patch("src.utils.identify_phone")
    def test_process_frame_with_phone_and_face(
        self,
        mock_identify_phone,
        mock_face_mesh_process,
        mock_draw_phone_alert,
        mock_draw_no_phone,
        mock_analyze_face,
        mock_thread,
    ):
        """Testa process_frame quando um telefone e um rosto são detectados."""
        # Configurar o mock para identificar_phone retornar uma caixa de telefone
        mock_identify_phone.return_value = [100, 100, 200, 200]

        # Configurar o mock para face_mesh.process retornar um rosto
        mock_face_mesh_process.return_value = MagicMock(
            multi_face_landmarks=[MagicMock()]
        )

        # Criar uma imagem dummy
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Configurar o mock para calcular_fps
        with patch("src.utils.calculate_fps", return_value=30):
            processed_img, current_time = process_frame(
                img, prev_frame_time=1000
            )

        mock_draw_phone_alert.assert_called_once_with(
            img, [100, 100, 200, 200]
        )
        mock_draw_no_phone.assert_not_called()
        mock_analyze_face.assert_called_once()
        mock_thread.assert_called_once()

    @patch("src.utils.draw_no_phone")
    @patch("src.utils.draw_phone_alert")
    @patch("src.utils.face_mesh.process")
    @patch("src.utils.identify_phone")
    def test_process_frame_no_phone_no_face(
        self,
        mock_identify_phone,
        mock_face_mesh_process,
        mock_draw_phone_alert,
        mock_draw_no_phone,
    ):
        """Testa process_frame quando nenhum telefone e nenhum rosto são detectados."""
        # Configurar o mock para identificar_phone retornar vazio
        mock_identify_phone.return_value = []

        # Configurar o mock para face_mesh.process retornar nenhum rosto
        mock_face_mesh_process.return_value = MagicMock(
            multi_face_landmarks=None
        )

        # Criar uma imagem dummy
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Configurar o mock para calcular_fps
        with patch("src.utils.calculate_fps", return_value=30):
            processed_img, current_time = process_frame(
                img, prev_frame_time=1000
            )

        mock_draw_phone_alert.assert_not_called()
        mock_draw_no_phone.assert_called_once_with(img)

    @patch("src.utils.cv2.rectangle")
    @patch("src.utils.cv2.putText")
    def test_draw_phone_alert(self, mock_putText, mock_rectangle):
        """Testa a função draw_phone_alert para desenhar corretamente os alertas de telefone."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox_phone = [100, 100, 200, 200]
        draw_phone_alert(img, bbox_phone)

        # Verificar se os retângulos foram desenhados corretamente
        mock_rectangle.assert_any_call(
            img, (100, 100), (405, 150), (0, 0, 255), -1
        )
        mock_rectangle.assert_any_call(
            img, (100, 100), (200, 200), (0, 0, 255), 5
        )

        # Verificar se o texto foi desenhado corretamente
        mock_putText.assert_called_once_with(
            img,
            "ALERT! PHONE",
            (105, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
        )

    @patch("src.utils.cv2.rectangle")
    @patch("src.utils.cv2.putText")
    def test_draw_no_phone(self, mock_putText, mock_rectangle):
        """Testa a função draw_no_phone para desenhar corretamente o indicador de ausência de telefone."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_no_phone(img)

        # Verificar se o retângulo foi desenhado corretamente
        mock_rectangle.assert_any_call(
            img, (100, 100), (370, 150), (0, 255, 0), -1
        )

        # Verificar se o texto foi desenhado corretamente
        mock_putText.assert_called_once_with(
            img,
            "NO PHONE",
            (105, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
        )

    @patch("src.utils.check_sleep_alarm")
    @patch("src.utils.display_sleep_indicator")
    @patch("src.utils.update_time_and_status")
    @patch("src.utils.determine_eye_situation")
    @patch("src.utils.draw_reference_circles")
    @patch("src.utils.draw_eye_points")
    def test_analyze_face(
        self,
        mock_draw_eye_points,
        mock_draw_reference_circles,
        mock_determine_eye_situation,
        mock_update_time_and_status,
        mock_display_sleep_indicator,
        mock_check_sleep_alarm,
    ):
        """Testa a função analyze_face para garantir que todas as etapas internas são executadas corretamente."""
        # Configurar mock para face.landmark
        face = MagicMock()
        face.landmark = [MagicMock(x=0.5, y=0.5) for _ in range(468)]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        w, h = 640, 480

        analyze_face(img, face, w, h)

        mock_draw_eye_points.assert_called_once_with(img, face, w, h)
        mock_draw_reference_circles.assert_called_once()
        mock_determine_eye_situation.assert_called_once()
        mock_update_time_and_status.assert_called_once()
        mock_display_sleep_indicator.assert_called_once_with(img)
        mock_check_sleep_alarm.assert_called_once()

    @patch("src.utils.winsound.Beep")
    def test_alarm(self, mock_beep):
        """Testa a função alarm para emitir corretamente o alarme de telefone."""
        global alarm_control
        # Garantir que alarm_control está False antes do alarme
        alarm_control = False

        # Executar a função alarm
        alarm()

        # Verificar se alarm_control foi atualizado corretamente
        self.assertTrue(alarm_control)

        # Verificar se winsound.Beep foi chamado corretamente
        mock_beep.assert_called_with(2000, 1000)

        # Após o sleep, alarm_control deve ser False
        self.assertFalse(alarm_control)

    @patch("src.utils.winsound.Beep")
    def test_sleep_alarm(self, mock_beep):
        """Testa a função sleep_alarm para emitir corretamente o alarme de sono."""
        global sleep_alarm_control
        # Garantir que sleep_alarm_control está False antes do alarme
        sleep_alarm_control = False

        # Executar a função sleep_alarm
        sleep_alarm()

        # Verificar se sleep_alarm_control foi atualizado corretamente
        self.assertTrue(sleep_alarm_control)

        # Verificar se winsound.Beep foi chamado duas vezes
        self.assertEqual(mock_beep.call_count, 2)
        mock_beep.assert_any_call(2000, 1000)

        # Após o sleep, sleep_alarm_control deve ser False
        self.assertFalse(sleep_alarm_control)

    def test_dummy(self):
        """Teste dummy para verificar se os testes estão sendo executados."""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
