import cv2
import mediapipe as mp
import time

# Inicializa o Mediapipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Inicializa a captura de vídeo da webcam.
cap = cv2.VideoCapture(0)

# Verifica se a webcam foi aberta corretamente.
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

print("Pressione 'q' para sair...")

prev_frame_time = 0
new_frame_time = 0

while True:
    # Lê o frame da webcam.
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        # Se estiver lendo de um arquivo, pode ser o fim.
        continue

    # Para melhorar o desempenho, opcionalmente marque a imagem como não gravável
    # para passá-la por referência.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Desenha as anotações da malha facial na imagem.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            # Poderíamos adicionar aqui o desenho de contornos específicos se necessário
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=drawing_spec)
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=drawing_spec)

    # Calcula e exibe o FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostra a imagem resultante.
    cv2.imshow('Pisk&Click - Detecção Facial (Mediapipe)', image)

    # Sai do loop se a tecla 'q' for pressionada.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas.
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print("Aplicação encerrada.")