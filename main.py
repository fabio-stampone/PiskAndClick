import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- Constantes e Configurações ---
# Índices dos landmarks faciais (Mediapipe Face Mesh com 478 landmarks)
NOSE_TIP_INDEX = 1
LEFT_EYE_LANDMARKS_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS_IDXS = [362, 385, 387, 263, 380, 373]

# Limiares e contadores para detecção de piscada
EAR_THRESHOLD = 0.20 # Ajustar experimentalmente
BLINK_CONSECUTIVE_FRAMES = 2

# Controle do Mouse
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Resolução da tela detectada: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Suavização do movimento do mouse
SMOOTHING_FACTOR = 0.3 # Valor entre 0 e 1. Menor = mais suave, mais lento. Maior = mais rápido, menos suave.
prev_mouse_x, prev_mouse_y = pyautogui.position() # Posição inicial

# Área de controle (região da imagem da câmera usada para mapear para a tela)
# Ajustar esses valores pode ser necessário dependendo da posição do rosto na câmera
CONTROL_AREA_X_MIN = 0.3 # 30% da largura da imagem
CONTROL_AREA_X_MAX = 0.7 # 70% da largura da imagem
CONTROL_AREA_Y_MIN = 0.3 # 30% da altura da imagem
CONTROL_AREA_Y_MAX = 0.7 # 70% da altura da imagem

# Inversão de eixos (opcional)
INVERT_X_AXIS = False
INVERT_Y_AXIS = False

# Contadores de frames para piscadas
left_blink_counter = 0
right_blink_counter = 0
left_blink_detected = False
right_blink_detected = False

# Debounce para cliques (evitar múltiplos cliques por uma piscada longa)
CLICK_DEBOUNCE_TIME = 0.5 # segundos
last_left_click_time = 0
last_right_click_time = 0

# --- Funções Auxiliares ---
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(eye_landmarks, image_shape):
    try:
        coords_points = np.array([(int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])) for landmark in eye_landmarks])
        p1, p2, p3, p4, p5, p6 = coords_points
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)
        horizontal_dist = calculate_distance(p1, p4)
        if horizontal_dist == 0:
            return 0.0
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception as e:
        # print(f"Erro ao calcular EAR: {e}") # Descomentar para debug
        return 0.0

def map_value(value, from_min, from_max, to_min, to_max):
    """Mapeia um valor de um intervalo para outro."""
    # Limita o valor ao intervalo de origem
    value = max(min(value, from_max), from_min)
    # Calcula a proporção
    from_span = from_max - from_min
    to_span = to_max - to_min
    value_scaled = float(value - from_min) / float(from_span)
    # Mapeia para o novo intervalo
    return to_min + (value_scaled * to_span)

# --- Inicialização ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam. O controle do mouse não funcionará.")
    use_dummy_image = True
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    image_height, image_width, _ = dummy_image.shape
else:
    use_dummy_image = False
    # Pega as dimensões reais da câmera
    ret, frame = cap.read()
    if ret:
        image_height, image_width, _ = frame.shape
        print(f"Resolução da câmera detectada: {image_width}x{image_height}")
    else:
        print("Erro ao ler o primeiro frame da câmera. Usando dimensões padrão.")
        image_height, image_width = 480, 640 # Dimensões padrão

print("Pressione 'q' para sair...")
print("Movimente o nariz para controlar o cursor.")
print("Pisque o olho esquerdo para clique esquerdo, olho direito para clique direito.")

prev_frame_time = 0
new_frame_time = 0

# --- Loop Principal ---
try:
    while True:
        current_time = time.time()
        if use_dummy_image:
            success = True
            image = dummy_image.copy()
            results = None
            time.sleep(0.1)
        else:
            success, image = cap.read()
            if not success:
                print("Ignorando frame vazio da câmera.")
                continue
            # Inverter a imagem horizontalmente para efeito espelho
            image = cv2.flip(image, 1)

        # Processamento com Mediapipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True
        # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convertido de volta se necessário

        nose_coords_norm = None
        left_ear = 0.0
        right_ear = 0.0
        left_blink_detected_this_frame = False
        right_blink_detected_this_frame = False

        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # --- Rastreamento do Nariz (Coordenadas Normalizadas) ---
            nose_tip = face_landmarks[NOSE_TIP_INDEX]
            nose_coords_norm = (nose_tip.x, nose_tip.y)
            nose_pixel_coords = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
            cv2.circle(image, nose_pixel_coords, 5, (0, 0, 255), -1)

            # --- Detecção de Piscada (EAR) ---
            left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS_IDXS]
            right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS_IDXS]
            left_ear = calculate_ear(left_eye_landmarks, (image_height, image_width))
            right_ear = calculate_ear(right_eye_landmarks, (image_height, image_width))

            # Lógica de detecção de piscada
            if left_ear < EAR_THRESHOLD:
                left_blink_counter += 1
            else:
                if left_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_left_click_time > CLICK_DEBOUNCE_TIME):
                    left_blink_detected_this_frame = True
                    left_blink_detected = True
                    last_left_click_time = current_time
                left_blink_counter = 0

            if right_ear < EAR_THRESHOLD:
                right_blink_counter += 1
            else:
                if right_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_right_click_time > CLICK_DEBOUNCE_TIME):
                    right_blink_detected_this_frame = True
                    right_blink_detected = True
                    last_right_click_time = current_time
                right_blink_counter = 0

            # Desenha landmarks dos olhos (para debug)
            # for idx in LEFT_EYE_LANDMARKS_IDXS:
            #     lm = face_landmarks[idx]
            #     cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 2, (0, 255, 0), -1)
            # for idx in RIGHT_EYE_LANDMARKS_IDXS:
            #     lm = face_landmarks[idx]
            #     cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 2, (0, 255, 0), -1)

        # --- Controle do Mouse ---
        if nose_coords_norm and not use_dummy_image:
            # Mapeia a posição normalizada do nariz para as coordenadas da tela
            target_x = map_value(nose_coords_norm[0], CONTROL_AREA_X_MIN, CONTROL_AREA_X_MAX, 0, SCREEN_WIDTH)
            target_y = map_value(nose_coords_norm[1], CONTROL_AREA_Y_MIN, CONTROL_AREA_Y_MAX, 0, SCREEN_HEIGHT)

            # Inverte eixos se necessário
            if INVERT_X_AXIS:
                target_x = SCREEN_WIDTH - target_x
            if INVERT_Y_AXIS:
                target_y = SCREEN_HEIGHT - target_y

            # Aplica suavização (média móvel exponencial)
            current_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING_FACTOR
            current_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING_FACTOR

            # Move o mouse
            try:
                pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))
                prev_mouse_x, prev_mouse_y = current_mouse_x, current_mouse_y
            except pyautogui.FailSafeException:
                print("FailSafe ativado (mouse movido para o canto). Encerrando...")
                break
            except Exception as e:
                print(f"Erro ao mover o mouse: {e}")
                # Pode ser necessário tratamento adicional dependendo do erro

        # --- Cliques do Mouse ---
        if left_blink_detected_this_frame and not use_dummy_image:
            try:
                pyautogui.click(button='left')
                print("Clique Esquerdo!")
                left_blink_detected = True # Para exibição
            except Exception as e:
                print(f"Erro ao clicar (esquerdo): {e}")

        if right_blink_detected_this_frame and not use_dummy_image:
            try:
                pyautogui.click(button='right')
                print("Clique Direito!")
                right_blink_detected = True # Para exibição
            except Exception as e:
                print(f"Erro ao clicar (direito): {e}")

        # --- Exibição de Informações ---
        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        prev_frame_time = new_frame_time

        cv2.putText(image, f"EAR Esq: {left_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"EAR Dir: {right_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        if left_blink_detected:
            cv2.putText(image, "CLIQUE ESQ", (image_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if right_blink_detected:
            cv2.putText(image, "CLIQUE DIR", (image_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Resetar estado visual de piscada após um tempo ou quando o olho abrir
        if left_ear >= EAR_THRESHOLD:
            left_blink_detected = False
        if right_ear >= EAR_THRESHOLD:
            right_blink_detected = False

        # Mostra a imagem
        try:
            cv2.imshow('Pisk&Click - Controle Facial', image)
        except cv2.error as e:
            if "display" in str(e).lower():
                if not use_dummy_image: # Só avisa se não estivermos no modo dummy
                    print("Aviso: Não foi possível exibir a janela (ambiente sem GUI?). Controle do mouse/clique ainda ativo.")
            else:
                print(f"Erro no cv2.imshow: {e}")

        # Sai do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupção pelo usuário.")

finally:
    # --- Finalização ---
    print("Encerrando aplicação...")
    if not use_dummy_image and cap.isOpened():
        cap.release()
    if 'cv2' in locals() and hasattr(cv2, 'destroyAllWindows'):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass # Ignora erro se não houver GUI
    if 'face_mesh' in locals():
        face_mesh.close()
    print("Aplicação encerrada.")