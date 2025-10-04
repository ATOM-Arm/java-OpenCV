_# 📌 Documentação - Detecção de Mãos com OpenCV em Java

Este código implementa um **sistema de detecção e reconhecimento de gestos manuais** utilizando a biblioteca **OpenCV (Java bindings)**.

Ele detecta a mão em tempo real via webcam, aplica processamento de imagem para segmentar a pele, extrai o contorno da mão e calcula **Convex Hull** e **Convexity Defects** para estimar a quantidade de dedos levantados.

---

## ⚙️ Dependências

- OpenCV (versão Java)
- Biblioteca `highgui` para exibição da imagem
- Webcam ou câmera disponível

---

## 🔄 Fluxo do Programa

1. **Captura de Vídeo**
    - Abre a webcam com `VideoCapture(0)`.
    - Lê frames em tempo real.

2. **Pré-processamento**
    - Espelha a imagem (modo selfie).
    - Aplica **GaussianBlur** para suavizar ruídos.
    - Converte o frame para o espaço de cor **YCrCb**.
    - Aplica máscara (`inRange`) para extrair tons de pele.
    - Realiza **operações morfológicas** (OPEN/CLOSE) para remover ruídos.
    - Suaviza com **MedianBlur**.

3. **Extração de Contornos**
    - Detecta os contornos com `Imgproc.findContours`.
    - Seleciona o **maior contorno** (mão principal).
    - Aproxima o contorno com `approxPolyDP`.

4. **Convex Hull e Defeitos de Convexidade**
    - Calcula **Convex Hull** da mão e desenha no frame.
    - Extrai **Convexity Defects**, que indicam "vales" entre os dedos.

5. **Contagem de Dedos**
    - Para cada defeito, calcula:
        - Profundidade (distância entre hull e contorno).
        - Ângulo formado entre pontos vizinhos.
    - Conta dedos quando:
        - `depth > 25` (dedo bem separado).
        - `angle < 85°` (dedos mais abertos).
    - Resultado final limitado a **5 dedos**.

6. **Classificação de Gestos**
    - Traduz número de dedos em um rótulo:
        - `0` → Fist (punho fechado)
        - `1` → 1 Finger
        - `2` → 2 Fingers
        - `3` → 3 Fingers
        - `4` → 4 Fingers
        - `5` → 5 Fingers

7. **Exibição**
    - Mostra o frame processado em uma janela (`HighGui.imshow`).
    - Sobrepõe texto com:
        - Quantidade de dedos
        - Gesto identificado

---

## 🛠️ Funções Auxiliares

### 🔹 `hullPointsFromIndices`
Converte os índices do **Convex Hull** em pontos reais do contorno.

### 🔹 `countFingers`
- Recebe os **defeitos de convexidade**.
- Verifica se cada vale corresponde a um dedo.
- Retorna número de dedos encontrados.

### 🔹 `calcAngle`
- Calcula ângulo entre três pontos usando **lei dos cossenos**.

### 🔹 `dist`
- Distância Euclidiana entre dois pontos.

### 🔹 `classifyGesture`
- Associa número de dedos a um gesto nomeado.

---

## 📊 Resultados Esperados_
