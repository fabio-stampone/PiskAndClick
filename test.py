import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
import sys
import traceback

# --- Constantes e Configurações ---
# Índices dos landmarks faciais (Mediapipe Face Mesh)
NOSE_TIP_INDEX = 1
LEFT_EYE_LANDMARKS_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS_IDXS = [362, 385, 387, 263, 380, 373]
MOUTH_TOP_INDEX = 13
MOUTH_BOTTOM_INDEX = 14
LEFT_EYE_CORNER_INDEX = 133
RIGHT_EYE_CORNER_INDEX = 362

# --- Parâmetros Ajustáveis (Otimização e Personalização) ---
AUTO_CALIBRATE_EAR = True
CALIBRATION_DURATION_OPEN = 5
CALIBRATION_DURATION_BLINK = 5
CALIBRATION_EAR_FACTOR = 0.35
REFINE_LANDMARKS = True
DEFAULT_EAR_THRESHOLD = 0.20
BLINK_CONSECUTIVE_FRAMES = 2
MOUTH_OPEN_RATIO_THRESHOLD = 0.6
MOUTH_OPEN_CONSECUTIVE_FRAMES = 3
SCROLL_AMOUNT = -60
SCROLL_DEBOUNCE_TIME = 0.3
TERMINATION_BOTH_EYES_CLOSED_DURATION = 2.0
TERMINATION_CONSECUTIVE_FRAMES = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING_FACTOR = 0.3
CONTROL_AREA_X_MIN, CONTROL_AREA_X_MAX = 0.3, 0.7
CONTROL_AREA_Y_MIN, CONTROL_AREA_Y_MAX = 0.3, 0.7
INVERT_X_AXIS, INVERT_Y_AXIS = False, False
CLICK_DEBOUNCE_TIME = 0.4
PROCESS_EVERY_N_FRAMES = 1

# --- Variáveis Globais e de Estado ---
calibration_state = "STARTING"
calibration_start_time = 0
open_ear_values = []
blink_ear_values = []
calibrated_ear_threshold = DEFAULT_EAR_THRESHOLD
frame_counter = 0
prev_frame_time = 0
fps_history = []
avg_fps = 30
left_blink_counter = 0
right_blink_counter = 0
left_blink_detected = False
right_blink_detected = False
mouth_open_counter = 0
both_eyes_closed_counter = 0
last_left_click_time = 0
last_right_click_time = 0
last_scroll_time = 0
last_termination_check_time = 0
prev_mouse_x, prev_mouse_y = pyautogui.position()

# --- Funções Auxiliares ---
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_landmark_coords(landmark, image_shape):
    return (int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0]))

def calculate_ear(eye_landmarks, image_shape):
    try:
        coords_points = [get_landmark_coords(lm, image_shape) for lm in eye_landmarks]
        p1, p2, p3, p4, p5, p6 = coords_points
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)
        horizontal_dist = calculate_distance(p1, p4)
        if horizontal_dist == 0: return 0.0
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception: return 0.0

def calculate_mouth_open_ratio(landmarks, image_shape):
    try:
        mouth_top_coords = get_landmark_coords(landmarks[MOUTH_TOP_INDEX], image_shape)
        mouth_bottom_coords = get_landmark_coords(landmarks[MOUTH_BOTTOM_INDEX], image_shape)
        left_eye_corner_coords = get_landmark_coords(landmarks[LEFT_EYE_CORNER_INDEX], image_shape)
        right_eye_corner_coords = get_landmark_coords(landmarks[RIGHT_EYE_CORNER_INDEX], image_shape)
        mouth_height = calculate_distance(mouth_top_coords, mouth_bottom_coords)
        eye_distance = calculate_distance(left_eye_corner_coords, right_eye_corner_coords)
        if eye_distance == 0: return 0.0
        ratio = mouth_height / eye_distance
        return ratio
    except Exception: return 0.0

def map_value(value, from_min, from_max, to_min, to_max):
    value = max(min(value, from_max), from_min)
    from_span = from_max - from_min
    to_span = to_max - to_min
    if from_span == 0: return to_min
    value_scaled = float(value - from_min) / float(from_span)
    return to_min + (value_scaled * to_span)

def update_fps(current_time):
    global prev_frame_time, fps_history, avg_fps
    fps = avg_fps # Default to average if calculation fails
    if prev_frame_time > 0:
        try:
            delta_time = current_time - prev_frame_time
            if delta_time > 0:
                fps = 1 / delta_time
                fps_history.append(fps)
                if len(fps_history) > 50:
                    fps_history.pop(0)
                if fps_history:
                    avg_fps = sum(fps_history) / len(fps_history)
            else:
                 fps = avg_fps # Use average if delta_time is zero or negative
        except ZeroDivisionError:
            fps = avg_fps
    prev_frame_time = current_time
    return fps

# --- Inicialização ---
print("Inicializando Pisk&Click v6...")
print(f"Resolução da tela: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=REFINE_LANDMARKS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

print("Abrindo a webcam...")
cap = cv2.VideoCapture(0)
image_height, image_width = 480, 640
use_dummy_image = False
if not cap.isOpened():
    print("Erro: Webcam não encontrada. Executando em modo dummy.")
    use_dummy_image = True
    dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
else:
    ret, frame = cap.read()
    if ret:
        image_height, image_width, _ = frame.shape
        print(f"Resolução da câmera: {image_width}x{image_height}")
    else:
        print("Erro ao ler frame. Usando dimensões padrão e modo dummy.")
        cap.release()
        use_dummy_image = True
        dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

if AUTO_CALIBRATE_EAR and not use_dummy_image:
    calibration_state = "CALIBRATING_OPEN_START"
    calibration_start_time = time.time()
else:
    calibration_state = "RUNNING"
    calibrated_ear_threshold = DEFAULT_EAR_THRESHOLD
    print(f"Calibração desligada/dummy. Usando EAR Threshold: {calibrated_ear_threshold:.3f}")

print("--- Pisk&Click Iniciado ---")
if calibration_state != "RUNNING":
    print("Iniciando calibração...")
else:
    print("Pressione 'q' ou feche ambos os olhos (~2s) para sair.")
    # ... (other instructions) ...

# --- Loop Principal ---
should_exit = False
try:
    while not should_exit:
        current_time = time.time()
        frame_counter += 1

        if use_dummy_image:
            success = True
            image = dummy_image.copy()
            results = None
            time.sleep(0.1)
        else:
            success, image = cap.read()
            if not success:
                print("Ignorando frame vazio.")
                continue
            image = cv2.flip(image, 1)

        process_this_frame = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)
        nose_coords_norm = None
        face_landmarks = None
        left_ear, right_ear, mouth_ratio = 0.0, 0.0, 0.0

        if process_this_frame:
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            image.flags.writeable = True

            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                nose_tip = face_landmarks[NOSE_TIP_INDEX]
                nose_coords_norm = (nose_tip.x, nose_tip.y)
                nose_pixel_coords = get_landmark_coords(nose_tip, (image_height, image_width))
                cv2.circle(image, nose_pixel_coords, 5, (0, 0, 255), -1)
                left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS_IDXS]
                right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS_IDXS]
                left_ear = calculate_ear(left_eye_landmarks, (image_height, image_width))
                right_ear = calculate_ear(right_eye_landmarks, (image_height, image_width))
                mouth_ratio = calculate_mouth_open_ratio(face_landmarks, (image_height, image_width))
            # else: # Face not detected in this frame
                # Ensure values are reset if needed, though they are reset at loop start
                # left_ear, right_ear, mouth_ratio = 0.0, 0.0, 0.0
                # nose_coords_norm = None
                # pass

        fps = update_fps(current_time)
        if TERMINATION_CONSECUTIVE_FRAMES == 0 and avg_fps > 0:
             TERMINATION_CONSECUTIVE_FRAMES = int(TERMINATION_BOTH_EYES_CLOSED_DURATION * avg_fps)
             if TERMINATION_CONSECUTIVE_FRAMES > 0: # Ensure it's positive
                print(f"FPS: {avg_fps:.1f}. Terminação: {TERMINATION_CONSECUTIVE_FRAMES} frames.")
             else:
                 TERMINATION_CONSECUTIVE_FRAMES = 1 # Failsafe if FPS is too low

        if calibration_state != "RUNNING":
            elapsed_time = current_time - calibration_start_time
            remaining_time = 0
            if calibration_state == "CALIBRATING_OPEN_START":
                print("CALIBRAÇÃO: Olhos ABERTOS...")
                calibration_state = "CALIBRATING_OPEN"
            if calibration_state == "CALIBRATING_OPEN":
                remaining_time = CALIBRATION_DURATION_OPEN - elapsed_time
                cv2.putText(image, f"OLHOS ABERTOS: {remaining_time:.1f}s", (50, image_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if face_landmarks and left_ear > 0 and right_ear > 0:
                    open_ear_values.append((left_ear + right_ear) / 2.0)
                if elapsed_time >= CALIBRATION_DURATION_OPEN:
                    calibration_state = "CALIBRATING_BLINK_START"
                    calibration_start_time = time.time()
            elif calibration_state == "CALIBRATING_BLINK_START":
                print("CALIBRAÇÃO: PISQUE agora...")
                calibration_state = "CALIBRATING_BLINK"
            elif calibration_state == "CALIBRATING_BLINK":
                remaining_time = CALIBRATION_DURATION_BLINK - elapsed_time
                cv2.putText(image, f"PISQUE AGORA: {remaining_time:.1f}s", (50, image_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if face_landmarks and left_ear > 0 and right_ear > 0:
                    if left_ear < (DEFAULT_EAR_THRESHOLD + 0.1) and right_ear < (DEFAULT_EAR_THRESHOLD + 0.1):
                         blink_ear_values.append((left_ear + right_ear) / 2.0)
                if elapsed_time >= CALIBRATION_DURATION_BLINK:
                    if open_ear_values and blink_ear_values:
                        avg_open_ear = sum(open_ear_values) / len(open_ear_values)
                        avg_min_blink_ear = sum(blink_ear_values) / len(blink_ear_values)
                        calibrated_ear_threshold = avg_min_blink_ear + (avg_open_ear - avg_min_blink_ear) * CALIBRATION_EAR_FACTOR
                        print(f"Calibração OK: EAR Aberto={avg_open_ear:.3f}, Piscada={avg_min_blink_ear:.3f}")
                        print(f"EAR Threshold Calibrado: {calibrated_ear_threshold:.3f}")
                    else:
                        calibrated_ear_threshold = DEFAULT_EAR_THRESHOLD
                        print("Calibração falhou. Usando threshold padrão.")
                    calibration_state = "RUNNING"
                    # Reset counters just before running state starts
                    left_blink_counter = 0
                    right_blink_counter = 0
                    mouth_open_counter = 0
                    both_eyes_closed_counter = 0
                    print("--- Controle Ativado ---")

        # --- Lógica Principal (RUNNING) ---
        elif calibration_state == "RUNNING" and not use_dummy_image:
            # This block now only depends on calibration_state and not use_dummy_image
            try: # Wrap the core logic in a try-except
                left_blink_detected_this_frame = False
                right_blink_detected_this_frame = False
                mouth_open_detected_this_frame = False
                terminate_detected_this_frame = False

                # Only perform gesture detection if face_landmarks were found in this processed frame
                if face_landmarks:
                    # Detecção de Piscada Esquerda
                    if left_ear < calibrated_ear_threshold:
                        left_blink_counter += 1
                    else:
                        if left_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_left_click_time > CLICK_DEBOUNCE_TIME):
                            left_blink_detected_this_frame = True
                            last_left_click_time = current_time
                        left_blink_counter = 0
                    # Detecção de Piscada Direita
                    if right_ear < calibrated_ear_threshold:
                        right_blink_counter += 1
                    else:
                        if right_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_right_click_time > CLICK_DEBOUNCE_TIME):
                            right_blink_detected_this_frame = True
                            last_right_click_time = current_time
                        right_blink_counter = 0
                    # Detecção de Boca Aberta
                    if mouth_ratio > MOUTH_OPEN_RATIO_THRESHOLD:
                        mouth_open_counter += 1
                    else:
                        if mouth_open_counter >= MOUTH_OPEN_CONSECUTIVE_FRAMES and (current_time - last_scroll_time > SCROLL_DEBOUNCE_TIME):
                            mouth_open_detected_this_frame = True
                            last_scroll_time = current_time
                        mouth_open_counter = 0
                    # Detecção de Gesto de Encerramento
                    if left_ear < calibrated_ear_threshold and right_ear < calibrated_ear_threshold:
                        both_eyes_closed_counter += 1
                    else:
                        both_eyes_closed_counter = 0
                    if TERMINATION_CONSECUTIVE_FRAMES > 0 and both_eyes_closed_counter >= TERMINATION_CONSECUTIVE_FRAMES:
                         terminate_detected_this_frame = True
                else: # face_landmarks is None, reset counters to avoid stale state issues
                    left_blink_counter = 0
                    right_blink_counter = 0
                    mouth_open_counter = 0
                    both_eyes_closed_counter = 0

                # --- Controle do Mouse (only if nose was detected) ---
                if nose_coords_norm:
                    target_x = map_value(nose_coords_norm[0], CONTROL_AREA_X_MIN, CONTROL_AREA_X_MAX, 0, SCREEN_WIDTH)
                    target_y = map_value(nose_coords_norm[1], CONTROL_AREA_Y_MIN, CONTROL_AREA_Y_MAX, 0, SCREEN_HEIGHT)
                    if INVERT_X_AXIS: target_x = SCREEN_WIDTH - target_x
                    if INVERT_Y_AXIS: target_y = SCREEN_HEIGHT - target_y
                    current_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING_FACTOR
                    current_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING_FACTOR
                    try:
                        pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))
                        prev_mouse_x, prev_mouse_y = current_mouse_x, current_mouse_y
                    except pyautogui.FailSafeException:
                        print("FailSafe ativado. Encerrando...")
                        should_exit = True
                    except Exception as e: print(f"Erro ao mover mouse: {e}")

                # --- Ações ---
                if left_blink_detected_this_frame:
                    try:
                        pyautogui.click(button='left')
                        print("Clique Esquerdo!")
                        left_blink_detected = True
                    except Exception as e: print(f"Erro clique esquerdo: {e}")
                if right_blink_detected_this_frame:
                    try:
                        pyautogui.click(button='right')
                        print("Clique Direito!")
                        right_blink_detected = True
                    except Exception as e: print(f"Erro clique direito: {e}")
                if mouth_open_detected_this_frame:
                    try:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        print(f"Rolagem: {SCROLL_AMOUNT}")
                    except Exception as e: print(f"Erro rolagem: {e}")
                if terminate_detected_this_frame:
                    print("Gesto de encerramento detectado. Encerrando...")
                    should_exit = True

            except Exception as e:
                print(f"!!! ERRO INESPERADO NO LOOP PRINCIPAL (RUNNING): {e} !!!")
                traceback.print_exc()
                # Continue running, but log the error
                pass

        # --- Exibição de Informações ---
        if calibration_state == "RUNNING":
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"EAR L: {left_ear:.2f} R: {right_ear:.2f} Thr: {calibrated_ear_threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(image, f"Boca: {mouth_ratio:.2f} Thr: {MOUTH_OPEN_RATIO_THRESHOLD:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if left_blink_detected: cv2.putText(image, "CLIQUE ESQ", (image_width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if right_blink_detected: cv2.putText(image, "CLIQUE DIR", (image_width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mouth_open_detected_this_frame: cv2.putText(image, "ROLAGEM", (image_width - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if both_eyes_closed_counter > 0 and TERMINATION_CONSECUTIVE_FRAMES > 0:
                 cv2.putText(image, f"SAINDO... {both_eyes_closed_counter}/{TERMINATION_CONSECUTIVE_FRAMES}", (image_width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if left_ear >= calibrated_ear_threshold: left_blink_detected = False
            if right_ear >= calibrated_ear_threshold: right_blink_detected = False

        try:
            cv2.imshow('Pisk&Click v6 - Controle Facial', image)
        except cv2.error as e:
            if "display" in str(e).lower():
                if not use_dummy_image and frame_counter % 120 == 0: print("Aviso: Sem GUI. Controle ativo.")
            else: print(f"Erro cv2.imshow: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tecla 'q' pressionada. Encerrando...")
            should_exit = True

except KeyboardInterrupt:
    print("\nInterrupção pelo usuário (Ctrl+C).")
finally:
    print("Encerrando aplicação...")
    if not use_dummy_image and cap.isOpened():
        cap.release()
        print("Webcam liberada.")
    if 'cv2' in locals() and hasattr(cv2, 'destroyAllWindows'):
        try: cv2.destroyAllWindows()
        except cv2.error: pass
    if 'face_mesh' in locals():
        face_mesh.close()
        print("Recursos do Mediapipe liberados.")
    print("Aplicação Pisk&Click encerrada.")
    # sys.exit() # Avoid calling sys.exit() within finally if possible