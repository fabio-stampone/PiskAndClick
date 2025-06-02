import cv2
import mediapipe as mp
import time
import math
import numpy as np

# --- Constantes e Configurações ---
# Índices dos landmarks faciais (Mediapipe Face Mesh com 478 landmarks)
# Nariz
NOSE_TIP_INDEX = 1

# Olho Esquerdo (para EAR)
LEFT_EYE_LANDMARKS_IDXS = [
    33,  # Canto direito
    160, # Pálpebra superior (interno)
    158, # Pálpebra superior (externo)
    133, # Canto esquerdo
    153, # Pálpebra inferior (externo)
    144, # Pálpebra inferior (interno)
]

# Olho Direito (para EAR)
RIGHT_EYE_LANDMARKS_IDXS = [
    362, # Canto esquerdo
    385, # Pálpebra superior (externo)
    387, # Pálpebra superior (interno)
    263, # Canto direito
    380, # Pálpebra inferior (interno)
    373, # Pálpebra inferior (externo)
]

# Limiares e contadores para detecção de piscada
EAR_THRESHOLD = 0.20 # Limiar de EAR para considerar o olho fechado (ajustar experimentalmente)
BLINK_CONSECUTIVE_FRAMES = 2 # Número de frames consecutivos com EAR abaixo do limiar para registrar piscada

# Contadores de frames para piscadas
left_blink_counter = 0
right_blink_counter = 0
left_blink_detected = False
right_blink_detected = False

# --- Funções Auxiliares ---
def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos 2D."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(eye_landmarks, image_shape):
    """Calcula o Eye Aspect Ratio (EAR) para um olho."""
    try:
        # Converte coordenadas normalizadas para coordenadas de pixel
        coords_points = np.array([(int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0])) for landmark in eye_landmarks])

        # Extrai os pontos P1 a P6
        p1, p2, p3, p4, p5, p6 = coords_points

        # Calcula as distâncias verticais
        vertical_dist1 = calculate_distance(p2, p6)
        vertical_dist2 = calculate_distance(p3, p5)

        # Calcula a distância horizontal
        horizontal_dist = calculate_distance(p1, p4)

        # Evita divisão por zero
        if horizontal_dist == 0:
            return 0.0

        # Calcula o EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except Exception as e:
        print(f"Erro ao calcular EAR: {e}")
        return 0.0

# --- Inicialização ---
# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    # Simulação: Criar uma imagem preta se não houver webcam
    # exit()
    use_dummy_image = True
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
else:
    use_dummy_image = False

print("Pressione 'q' para sair...")

prev_frame_time = 0
new_frame_time = 0

# --- Loop Principal ---
while True:
    if use_dummy_image:
        success = True
        image = dummy_image.copy()
        # Simular um rosto detectado para continuar o desenvolvimento
        # (Isto é apenas para desenvolvimento sem webcam, não funcionará na prática)
        # Você precisaria fornecer landmarks simulados aqui se quisesse testar a lógica EAR/Nose
        results = None # Sem resultados reais
        time.sleep(0.1) # Simular taxa de frames
    else:
        success, image = cap.read()
        if not success:
            print("Ignorando frame vazio da câmera.")
            continue

    image_height, image_width, _ = image.shape

    # Processamento com Mediapipe
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True
    # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convertido de volta abaixo se necessário

    nose_coords = None
    left_ear = 0.0
    right_ear = 0.0
    left_blink_detected_this_frame = False
    right_blink_detected_this_frame = False

    if results and results.multi_face_landmarks:
        # Assumindo apenas um rosto (max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- Rastreamento do Nariz ---
        nose_tip = face_landmarks[NOSE_TIP_INDEX]
        nose_coords = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
        # Desenha um círculo na ponta do nariz
        cv2.circle(image, nose_coords, 5, (0, 0, 255), -1)

        # --- Detecção de Piscada (EAR) ---
        # Extrai landmarks dos olhos
        left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE_LANDMARKS_IDXS]
        right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE_LANDMARKS_IDXS]

        # Calcula EAR para cada olho
        left_ear = calculate_ear(left_eye_landmarks, (image_height, image_width))
        right_ear = calculate_ear(right_eye_landmarks, (image_height, image_width))

        # Verifica piscada do olho esquerdo
        if left_ear < EAR_THRESHOLD:
            left_blink_counter += 1
        else:
            if left_blink_counter >= BLINK_CONSECUTIVE_FRAMES:
                left_blink_detected_this_frame = True # Registra a piscada
                left_blink_detected = True # Mantém estado para exibição
            left_blink_counter = 0

        # Verifica piscada do olho direito
        if right_ear < EAR_THRESHOLD:
            right_blink_counter += 1
        else:
            if right_blink_counter >= BLINK_CONSECUTIVE_FRAMES:
                right_blink_detected_this_frame = True # Registra a piscada
                right_blink_detected = True # Mantém estado para exibição
            right_blink_counter = 0

        # Desenha a malha facial (opcional, pode consumir recursos)
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=results.multi_face_landmarks[0],
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_spec)

        # Desenha landmarks dos olhos (para debug)
        for idx in LEFT_EYE_LANDMARKS_IDXS:
            lm = face_landmarks[idx]
            cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 2, (0, 255, 0), -1)
        for idx in RIGHT_EYE_LANDMARKS_IDXS:
            lm = face_landmarks[idx]
            cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 2, (0, 255, 0), -1)

    # --- Exibição de Informações ---
    # Calcula e exibe o FPS
    new_frame_time = time.time()
    if prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    prev_frame_time = new_frame_time

    # Exibe EAR
    cv2.putText(image, f"EAR Esq: {left_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"EAR Dir: {right_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Exibe status da piscada
    if left_blink_detected:
        cv2.putText(image, "PISCADA ESQUERDA!", (image_width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # Resetar estado após exibição (ou manter por alguns frames)
        # left_blink_detected = False
    if right_blink_detected:
        cv2.putText(image, "PISCADA DIREITA!", (image_width - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        # Resetar estado após exibição (ou manter por alguns frames)
        # right_blink_detected = False

    # Resetar estado geral de piscada detectada para a próxima detecção
    # (Isto é diferente de left_blink_detected_this_frame)
    if left_ear >= EAR_THRESHOLD:
        left_blink_detected = False
    if right_ear >= EAR_THRESHOLD:
        right_blink_detected = False

    # Mostra a imagem resultante.
    # A exibição pode falhar no sandbox, mas o código de lógica é o importante
    try:
        cv2.imshow('Pisk&Click - Rastreamento Facial', image)
    except cv2.error as e:
        if "display" in str(e).lower():
            print("Aviso: Não foi possível exibir a janela (ambiente sem GUI?). Continuando a execução da lógica.")
        else:
            print(f"Erro no cv2.imshow: {e}")
            # break # Descomente para parar em outros erros de imshow

    # Sai do loop se a tecla 'q' for pressionada.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Finalização ---
if not use_dummy_image:
    cap.release()
if 'cv2' in locals() and hasattr(cv2, 'destroyAllWindows'):
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        print("Aviso: Não foi possível destruir janelas (ambiente sem GUI?).")
face_mesh.close()

print("Aplicação encerrada.")