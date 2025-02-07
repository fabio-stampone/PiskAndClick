import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
from threading import Thread
from settings import SettingsWindow  # Importa a classe do outro arquivo

# Configurações do mouse
MOUSE_SPEED = 0.01
MOUSE_DPI_SCALE = 1.0
CURSOR_SIZE = 5


# Inicializa o FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# ---------- CONFIGURAÇÕES ----------
EAR_THRESHOLD = 0.23
BOTH_CLOSED_FRAMES = 3
nose_history = deque(maxlen=3)


# Estado da janela de configurações
settings_window_open = False


def eye_aspect_ratio(eye_points):
    vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal)


def update_mouse_settings(speed, dpi, cursor_size):
    """
    Callback para aplicar as configurações do mouse.
    """
    global MOUSE_SPEED, MOUSE_DPI_SCALE, CURSOR_SIZE
    MOUSE_SPEED = speed
    MOUSE_DPI_SCALE = dpi
    CURSOR_SIZE = cursor_size
    print(f"Configurações Atualizadas: Velocidade={speed}, DPI={dpi}, Cursor={cursor_size}")


def open_settings():
    """
    Abre a janela de configurações em uma thread separada.
    """
    global settings_window_open
    if settings_window_open:
        return
    settings_window_open = True

    def show_window():
        global settings_window_open
        settings = SettingsWindow(MOUSE_SPEED, MOUSE_DPI_SCALE, CURSOR_SIZE, update_mouse_settings)
        settings.show()
        settings_window_open = False

    Thread(target=show_window).start()


cv2.namedWindow('Facial Mouse', cv2.WINDOW_NORMAL)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Processa com MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

        nose = landmarks[4]
        nose_history.append((nose[0], nose[1]))
        smoothed_nose = np.mean(nose_history, axis=0)
        nose_x = int(smoothed_nose[0] * screen_w * MOUSE_DPI_SCALE)
        nose_y = int(smoothed_nose[1] * screen_h * MOUSE_DPI_SCALE)
        pyautogui.moveTo(nose_x, nose_y, duration=MOUSE_SPEED)

        # Detecta se ambos os olhos estão fechados
        both_closed = True  # Substitua pela lógica de detecção de piscadas
        if both_closed:
            open_settings()

    cv2.circle(image, (nose_x, nose_y), CURSOR_SIZE, (255, 0, 0), -1)
    cv2.imshow('Facial Mouse', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
