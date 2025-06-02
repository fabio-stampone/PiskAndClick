import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- Constantes e Configurações ---
# Índices dos landmarks faciais (Mediapipe Face Mesh)
# Consulte: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
NOSE_TIP_INDEX = 1
LEFT_EYE_LANDMARKS_IDXS = [33, 160, 158, 133, 153, 144] # Pontos para cálculo do EAR esquerdo
RIGHT_EYE_LANDMARKS_IDXS = [362, 385, 387, 263, 380, 373] # Pontos para cálculo do EAR direito

# --- Parâmetros Ajustáveis (Otimização e Personalização) ---

# Qualidade vs Desempenho Mediapipe
# refine_landmarks=True: Melhora a precisão dos landmarks dos lábios, olhos e íris, mas aumenta a carga computacional.
# Defina como False para melhor desempenho em hardware mais lento.
REFINE_LANDMARKS = True

# Limiares e Contadores para Detecção de Piscada
# EAR_THRESHOLD: Eye Aspect Ratio (EAR) abaixo do qual o olho é considerado fechado.
# *Ajuste Experimentalmente*: Este valor é sensível à pessoa, câmera e iluminação.
# Valores comuns ficam entre 0.18 e 0.25. Comece com 0.20 e ajuste.
EAR_THRESHOLD = 0.20
# BLINK_CONSECUTIVE_FRAMES: Número de frames consecutivos que o EAR deve estar abaixo do limiar para registrar uma piscada.
# Ajuda a evitar falsos positivos por movimentos rápidos. 2 ou 3 é geralmente um bom começo.
BLINK_CONSECUTIVE_FRAMES = 2

# Controle do Mouse
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Resolução da tela detectada: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Suavização do Movimento do Mouse
# SMOOTHING_FACTOR: Controla a suavidade do movimento do cursor (média móvel exponencial).
# Valor entre 0 e 1. Menor = mais suave, mas com mais atraso. Maior = mais rápido, menos suave.
# Ajuste para encontrar um equilíbrio confortável.
SMOOTHING_FACTOR = 0.3
prev_mouse_x, prev_mouse_y = pyautogui.position()

# Área de Controle do Rosto (Mapeamento Câmera -> Tela)
# Define a região da imagem da câmera (em proporção 0.0 a 1.0) que será mapeada para toda a tela.
# Ajuste esses valores se o cursor não alcançar as bordas da tela ou se mover muito rápido/devagar.
# Ex: Aumentar a diferença entre MIN e MAX torna o movimento mais sensível.
# Ex: Diminuir a diferença torna o movimento menos sensível.
CONTROL_AREA_X_MIN = 0.3 # 30% da largura da imagem (borda esquerda da área de controle)
CONTROL_AREA_X_MAX = 0.7 # 70% da largura da imagem (borda direita da área de controle)
CONTROL_AREA_Y_MIN = 0.3 # 30% da altura da imagem (borda superior da área de controle)
CONTROL_AREA_Y_MAX = 0.7 # 70% da altura da imagem (borda inferior da área de controle)

# Inversão de Eixos (Opcional)
# Defina como True se o movimento do cursor parecer invertido no eixo X ou Y.
INVERT_X_AXIS = False
INVERT_Y_AXIS = False

# Debounce para Cliques
# Evita múltiplos cliques registrados por uma única piscada mais longa.
CLICK_DEBOUNCE_TIME = 0.5 # Tempo mínimo (segundos) entre cliques do mesmo botão.
last_left_click_time = 0
last_right_click_time = 0

# Otimização de Desempenho (Opcional)
# PROCESS_EVERY_N_FRAMES: Processa o rosto apenas a cada N frames. Aumenta o FPS, mas introduz algum atraso.
# 1 = processa todos os frames. 2 = processa metade, etc.
PROCESS_EVERY_N_FRAMES = 1
frame_counter = 0

# --- Contadores e Estados Internos ---
left_blink_counter = 0
right_blink_counter = 0
left_blink_detected = False # Estado visual para feedback na tela
right_blink_detected = False # Estado visual para feedback na tela

# --- Funções Auxiliares ---
def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos 2D."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(eye_landmarks, image_shape):
    """Calcula o Eye Aspect Ratio (EAR) para um olho."""
    try:
        # Converte coordenadas normalizadas para coordenadas de pixel
        coords_points = np.array([(int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])) for landmark in eye_landmarks])
        p1, p2, p3, p4, p5, p6 = coords_points
        # Calcula as distâncias verticais e horizontal
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)
        horizontal_dist = calculate_distance(p1, p4)
        if horizontal_dist == 0: return 0.0
        # Calcula o EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception as e:
        # print(f"Erro ao calcular EAR: {e}") # Descomentar para debug
        return 0.0

def map_value(value, from_min, from_max, to_min, to_max):
    """Mapeia um valor de um intervalo para outro, com clamping."""
    value = max(min(value, from_max), from_min)
    from_span = from_max - from_min
    to_span = to_max - to_min
    if from_span == 0: return to_min # Evita divisão por zero
    value_scaled = float(value - from_min) / float(from_span)
    return to_min + (value_scaled * to_span)

# --- Inicialização ---
print("Inicializando Mediapipe Face Mesh...")
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
image_height, image_width = 480, 640 # Valores padrão
use_dummy_image = False
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam. O controle do mouse não funcionará.")
    use_dummy_image = True
    dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
else:
    ret, frame = cap.read()
    if ret:
        image_height, image_width, _ = frame.shape
        print(f"Resolução da câmera detectada: {image_width}x{image_height}")
    else:
        print("Erro ao ler o primeiro frame da câmera. Usando dimensões padrão.")
        cap.release() # Libera se a leitura inicial falhou
        use_dummy_image = True
        dummy_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

print("--- Pisk&Click Iniciado ---")
print("Pressione 'q' para sair.")
print("Movimente o nariz para controlar o cursor.")
print("Pisque o olho esquerdo para clique esquerdo, olho direito para clique direito.")
print(f"Ajustes Atuais: EAR Thresh={EAR_THRESHOLD}, Smooth={SMOOTHING_FACTOR}, Refine={REFINE_LANDMARKS}")

prev_frame_time = 0
new_frame_time = 0

# --- Loop Principal ---
try:
    while True:
        current_time = time.time()
        frame_counter += 1

        # Leitura do Frame
        if use_dummy_image:
            success = True
            image = dummy_image.copy()
            results = None # Sem resultados em modo dummy
            time.sleep(0.1) # Simula taxa de frames
        else:
            success, image = cap.read()
            if not success:
                print("Ignorando frame vazio da câmera.")
                continue
            # Inverter a imagem horizontalmente (efeito espelho)
            image = cv2.flip(image, 1)

        # Otimização: Processar apenas a cada N frames
        process_this_frame = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)

        nose_coords_norm = None
        left_ear = 0.0
        right_ear = 0.0
        left_blink_detected_this_frame = False
        right_blink_detected_this_frame = False

        if process_this_frame:
            # Processamento com Mediapipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            image.flags.writeable = True
            # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convertido de volta se necessário

            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # --- Rastreamento do Nariz (Coordenadas Normalizadas) ---
                nose_tip = face_landmarks[NOSE_TIP_INDEX]
                nose_coords_norm = (nose_tip.x, nose_tip.y)
                nose_pixel_coords = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
                cv2.circle(image, nose_pixel_coords, 5, (0, 0, 255), -1) # Desenha ponto no nariz

                # --- Detecção de Piscada (EAR) ---
                left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS_IDXS]
                right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS_IDXS]
                left_ear = calculate_ear(left_eye_landmarks, (image_height, image_width))
                right_ear = calculate_ear(right_eye_landmarks, (image_height, image_width))

                # Lógica de detecção de piscada com debounce
                if left_ear < EAR_THRESHOLD:
                    left_blink_counter += 1
                else:
                    if left_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_left_click_time > CLICK_DEBOUNCE_TIME):
                        left_blink_detected_this_frame = True
                        last_left_click_time = current_time
                    left_blink_counter = 0

                if right_ear < EAR_THRESHOLD:
                    right_blink_counter += 1
                else:
                    if right_blink_counter >= BLINK_CONSECUTIVE_FRAMES and (current_time - last_right_click_time > CLICK_DEBOUNCE_TIME):
                        right_blink_detected_this_frame = True
                        last_right_click_time = current_time
                    right_blink_counter = 0
            # else: # Se nenhum rosto for detectado neste frame
                # nose_coords_norm = None # Garante que o mouse não se mova
                # Opcional: resetar contadores de piscada se o rosto for perdido?
                # left_blink_counter = 0
                # right_blink_counter = 0

        # --- Controle do Mouse (Executa mesmo se o frame não foi processado, usando a última posição válida) ---
        if nose_coords_norm and not use_dummy_image:
            # Mapeia a posição normalizada do nariz para as coordenadas da tela
            target_x = map_value(nose_coords_norm[0], CONTROL_AREA_X_MIN, CONTROL_AREA_X_MAX, 0, SCREEN_WIDTH)
            target_y = map_value(nose_coords_norm[1], CONTROL_AREA_Y_MIN, CONTROL_AREA_Y_MAX, 0, SCREEN_HEIGHT)

            if INVERT_X_AXIS: target_x = SCREEN_WIDTH - target_x
            if INVERT_Y_AXIS: target_y = SCREEN_HEIGHT - target_y

            # Aplica suavização
            current_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING_FACTOR
            current_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING_FACTOR

            # Move o mouse
            try:
                # pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y), duration=0) # duration=0 é mais rápido
                pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))
                prev_mouse_x, prev_mouse_y = current_mouse_x, current_mouse_y
            except pyautogui.FailSafeException:
                print("FailSafe ativado (mouse movido para o canto). Encerrando...")
                break
            except Exception as e:
                print(f"Erro ao mover o mouse: {e}")

        # --- Cliques do Mouse ---
        if left_blink_detected_this_frame and not use_dummy_image:
            try:
                pyautogui.click(button='left')
                print("Clique Esquerdo!")
                left_blink_detected = True # Ativa feedback visual
            except Exception as e:
                print(f"Erro ao clicar (esquerdo): {e}")

        if right_blink_detected_this_frame and not use_dummy_image:
            try:
                pyautogui.click(button='right')
                print("Clique Direito!")
                right_blink_detected = True # Ativa feedback visual
            except Exception as e:
                print(f"Erro ao clicar (direito): {e}")

        # --- Exibição de Informações (Atualiza a cada frame) ---
        # Calcula FPS
        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        prev_frame_time = new_frame_time

        # Exibe EAR
        cv2.putText(image, f"EAR Esq: {left_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"EAR Dir: {right_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Exibe feedback visual de clique
        if left_blink_detected:
            cv2.putText(image, "CLIQUE ESQ", (image_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if right_blink_detected:
            cv2.putText(image, "CLIQUE DIR", (image_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Resetar estado visual de piscada se o olho reabriu
        if left_ear >= EAR_THRESHOLD: left_blink_detected = False
        if right_ear >= EAR_THRESHOLD: right_blink_detected = False

        # Mostra a imagem
        try:
            cv2.imshow('Pisk&Click - Controle Facial', image)
        except cv2.error as e:
            if "display" in str(e).lower():
                if not use_dummy_image and frame_counter % 60 == 0: # Avisa periodicamente se não houver GUI
                    print("Aviso: Não foi possível exibir a janela (ambiente sem GUI?). Controle do mouse/clique ainda ativo.")
            else:
                print(f"Erro no cv2.imshow: {e}")

        # Condição de Saída
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tecla 'q' pressionada. Encerrando...")
            break

except KeyboardInterrupt:
    print("\nInterrupção pelo usuário (Ctrl+C).")

finally:
    # --- Finalização Limpa ---
    print("Encerrando aplicação...")
    if not use_dummy_image and cap.isOpened():
        cap.release()
        print("Webcam liberada.")
    if 'cv2' in locals() and hasattr(cv2, 'destroyAllWindows'):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass # Ignora erro se não houver GUI
    if 'face_mesh' in locals():
        face_mesh.close()
        print("Recursos do Mediapipe liberados.")
    print("Aplicação Pisk&Click encerrada.")