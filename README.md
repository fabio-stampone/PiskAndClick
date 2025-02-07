# PiskAndClick

O **PiskAndClick** é uma aplicação que permite **controlar o mouse** usando a **ponta do seu nariz** e **realizar cliques** (esquerdo ou direito) piscando os olhos. Ele utiliza [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html) para rastrear pontos do rosto em tempo real e o [PyAutoGUI](https://pyautogui.readthedocs.io/) para interagir com o mouse do sistema.

---

## Sumário

- [Como Funciona](#como-funciona)  
- [Funcionalidades](#funcionalidades)  
- [Pré-Requisitos e Instalação](#pré-requisitos-e-instalação)  
- [Modo de Uso](#modo-de-uso)  
- [Calibração](#calibração)  
- [Dicas e Observações](#dicas-e-observações)  
- [Gerando .exe (Opcional)](#gerando-exe-opcional)  
- [Créditos](#créditos)

---

## Como Funciona

1. A webcam capta imagens em tempo real e o **MediaPipe FaceMesh** detecta pontos-chave (landmarks) do rosto.  
2. O software identifica a **ponta do nariz** (landmark 4) e converte essa posição para coordenadas de tela, movendo assim o cursor via PyAutoGUI.  
3. Para detectar piscadas, o projeto calcula a taxa de aspecto do olho (_Eye Aspect Ratio_ - EAR).  
   - Quando o EAR fica abaixo de um certo limite (_threshold_), considera-se o olho “fechado”.  
   - Dependendo de qual olho (esquerdo, direito ou ambos) estiver fechado por alguns frames, executa diferentes ações (cliques ou comandos).

---

## Funcionalidades

- **Mover o Cursor com o Nariz**  
  Mova seu rosto para direcionar o cursor na tela.  
- **Clique Esquerdo ou Direito ao Piscar**  
  - **Olho esquerdo**: executa clique esquerdo.  
  - **Olho direito**: executa clique direito.  
- **Calibração Automática**  
  Garante que o sistema ajuste o limiar de piscada (`EAR_THRESHOLD`) de acordo com as suas características faciais.  
- **Botão de Calibração Desenhado na Tela**  
  Ao posicionar o nariz sobre o botão (desenhado pela interface do OpenCV) e piscar, inicia-se o processo de calibração.

---

## Pré-Requisitos e Instalação

1. **Webcam** (interna ou externa) funcionando.  
2. **Python 3.7+** instalado (caso vá rodar o script diretamente em Python).  
3. Dependências utilizadas:
   - [OpenCV-Python](https://pypi.org/project/opencv-python/)  
   - [Mediapipe](https://pypi.org/project/mediapipe/)  
   - [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)  
   - [NumPy](https://numpy.org/)  
   - [collections (deque)](https://docs.python.org/3/library/collections.html) – já parte da biblioteca padrão Python.

### Instalação das Dependências (em Python)

```bash
pip install opencv-python mediapipe pyautogui numpy
