# 📌 Documentação - Detecção de Mãos com OpenCV em Java

Este projeto implementa um **sistema de detecção e reconhecimento de gestos manuais** utilizando a biblioteca **OpenCV (Java bindings)**.

O sistema detecta a mão em tempo real via webcam, segmenta a pele, extrai o contorno da mão e calcula **Convex Hull** e **Convexity Defects** para estimar a quantidade de dedos levantados. Além disso, registra métricas de desempenho e salva snapshots e logs em CSV.

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
        - `1` → Thumbs Up ou Index Up
        - `2` → Peace ou Rock
        - `3` → OK ou 3 Fingers
        - `4` → 4 Fingers
        - `5` → Palm

7. **Exibição e Registro**
    - Mostra o frame processado em uma janela (`HighGui.imshow`).
    - Sobrepõe texto com:
        - Quantidade de dedos
        - Gesto identificado
    - Salva snapshots ao pressionar a tecla `2`.
    - Registra métricas de cada frame em um arquivo CSV:
        - Número do frame, dedos detectados, área do contorno, centro da mão, número de defeitos, ângulo médio, FPS, uso de memória, uso de CPU, gesto identificado.

---

## 🛠️ Funções Auxiliares

### 🔹 `hullPointsFromIndices`
Converte os índices do **Convex Hull** em pontos reais do contorno.

### 🔹 `countFingers`
- Recebe os **defeitos de convexidade**.
- Verifica se cada vale corresponde a um dedo.
- Retorna número de dedos encontrados e ângulo médio.

### 🔹 `calcAngle`
- Calcula ângulo entre três pontos usando **lei dos cossenos**.

### 🔹 `dist`
- Distância Euclidiana entre dois pontos.

### 🔹 `classifyGesture`
- Associa número de dedos e características do contorno a um gesto nomeado.

### 🔹 `saveImage`
- Salva o frame atual como imagem PNG.

---

## 📊 Resultados Esperados

- Detecção em tempo real de gestos manuais.
- Classificação automática de gestos comuns (Fist, Palm, Thumbs Up, Peace, OK, etc).
- Registro de métricas de desempenho e uso de recursos.
- Possibilidade de salvar imagens para análise posterior.

---

## 🧩 Estrutura do Código (Resumo)

```java
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class handDetector {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    static String PATH = "src/src/handDetector/reports/";

    public static void main(String[] args) {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Erro ao abrir a webcam.");
            return;
        }

        Mat frame = new Mat();
        Mat mask = new Mat();
        Mat hierarchy = new Mat();

        // Inicializa o CSV
        try (PrintWriter csvWriter = new PrintWriter(new FileWriter(PATH + "csvs/performance"+ (int) (Math.random() * 1000) + ".csv"))) {
            csvWriter.println("frame,fingers,maxContourArea,centerX,centerY,convexDefects,avgAngle,fps,usedMemoryMB,cpuLoad,gesture");

            int frameNumber = 0; // Add this before the loop

            while (true) {
                long startTime = System.currentTimeMillis();
                if (!camera.read(frame) || frame.empty()) break;

                // Espelha a imagem (modo selfie)
                Core.flip(frame, frame, 1);

                // Suaviza ruídos
                Imgproc.GaussianBlur(frame, frame, new Size(5, 5), 0);

                // Converte para YCrCb e aplica máscara para tons de pele
                Mat ycrcb = new Mat();
                Imgproc.cvtColor(frame, ycrcb, Imgproc.COLOR_BGR2YCrCb);
                Core.inRange(ycrcb, new Scalar(0, 133, 77), new Scalar(255, 173, 127), mask);

                // Limpeza morfológica
                Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
                Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
                Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

                // Remoção de pequenos ruídos
                Imgproc.medianBlur(mask, mask, 5);

                // Encontra contornos
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

                double maxArea = 5000; // aumenta o filtro para ignorar pequenos contornos
                int index = -1;
                for (int i = 0; i < contours.size(); i++) {
                    double area = Imgproc.contourArea(contours.get(i));
                    if (area > maxArea) {
                        maxArea = area;
                        index = i;
                    }
                }

                int fingers = 0;
                String gesture = "";

                FingerData fingerData = new FingerData(0, 0);
                if (index != -1) {
                    MatOfPoint contour = contours.get(index);

                    // Aproximação poligonal
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                    Imgproc.approxPolyDP(contour2f, contour2f, 3, true);
                    MatOfPoint approxContour = new MatOfPoint();
                    contour2f.convertTo(approxContour, CvType.CV_32S);

                    Imgproc.drawContours(frame, contours, index, new Scalar(0, 255, 0), 2);

                    // Convex Hull
                    MatOfInt hull = new MatOfInt();
                    Imgproc.convexHull(approxContour, hull);
                    MatOfPoint hullPoints = hullPointsFromIndices(approxContour, hull);
                    List<MatOfPoint> hullList = new ArrayList<>();
                    hullList.add(hullPoints);
                    Imgproc.drawContours(frame, hullList, 0, new Scalar(255, 0, 0), 2);

                    // Convexity Defects
                    MatOfInt4 defects = new MatOfInt4();
                    Imgproc.convexityDefects(approxContour, hull, defects);

                    fingerData = countFingers(defects, approxContour);

                    // Atualiza o gesto
                    gesture = classifyGesture(fingerData.count, approxContour, defects);

                    // cálculo do centro
                    Moments m = Imgproc.moments(contour);
                    double cx = m.get_m10() / m.get_m00();
                    double cy = m.get_m01() / m.get_m00();
                    maxArea = Imgproc.contourArea(contour);
                    int convexDefects = (int) defects.total(); // ou somente os válidos

                    long endTime = System.currentTimeMillis();
                    double fps = 1000.0 / (endTime - startTime);

                    Runtime runtime = Runtime.getRuntime();
                    double usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024.0 / 1024.0;

                    // CPU load
                    com.sun.management.OperatingSystemMXBean osBean =
                            (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
                    double cpuLoad = osBean.getProcessCpuLoad();

                    // Escreve no CSV
                    csvWriter.printf(Locale.US,"%d,%d,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.4f,%s%n",
                            frameNumber++,
                            fingerData.count,
                            maxArea,
                            cx,
                            cy,
                            convexDefects,
                            fingerData.avgAngle,
                            fps,
                            usedMemoryMB,
                            cpuLoad,
                            gesture
                    );
                    csvWriter.flush();

                    // Mostra na tela
                    Imgproc.putText(frame,
                            "Dedos: " + fingerData.count + " - " + gesture,
                            new Point(20, 40),
                            Imgproc.FONT_HERSHEY_SIMPLEX,
                            1.0, new Scalar(0, 255, 0), 2);
                }

                // Mostra apenas uma janela com o resultado final
                HighGui.imshow("Detecção de Mão", frame);

                int key = HighGui.waitKey(100) & 0xFF; // aumenta o tempo de espera

                if (key == 27) {
                    break;
                } else if (key == 50) {
                    String filename = String.format(
                            PATH+ "media/hand_snapshot_%03d.png",
                            (int) (Math.random() * 1000)
                    );
                    saveImage(frame, filename);
                    System.out.println("📸 Imagem salva como " + filename);
                }

            }
        } catch (IOException e) {
            System.out.println("Erro ao criar arquivo CSV: " + e.getMessage());
            e.printStackTrace();
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }

    // ---------------- Funções auxiliares ----------------

    private static MatOfPoint hullPointsFromIndices(MatOfPoint contour, MatOfInt hull) {
        Point[] contourPts = contour.toArray();
        int[] hullIdx = hull.toArray();
        Point[] hullPts = new Point[hullIdx.length];
        for (int i = 0; i < hullIdx.length; i++) hullPts[i] = contourPts[hullIdx[i]];
        MatOfPoint mop = new MatOfPoint();
        mop.fromArray(hullPts);
        return mop;
    }

    private static FingerData countFingers(MatOfInt4 defects, MatOfPoint contour) {
        if (defects.empty()) return new FingerData(0, 0);
        int[] arr = defects.toArray();
        Point[] points = contour.toArray();
        int count = 0;
        double sumAngle = 0;
        int validDefects = 0;

        for (int i = 0; i < arr.length; i += 4) {
            int startIdx = arr[i];
            int endIdx = arr[i + 1];
            int farIdx = arr[i + 2];
            float depth = arr[i + 3] / 256.0f;
            if (depth > 25) {
                double angle = calcAngle(points[startIdx], points[farIdx], points[endIdx]);
                if (angle < 85) {
                    count++;
                    sumAngle += angle;
                    validDefects++;
                }
            }
        }
        double avgAngle = validDefects > 0 ? sumAngle / validDefects : 0;
        return new FingerData(Math.min(5, count + 1), avgAngle);
    }

    private static double calcAngle(Point a, Point b, Point c) {
        double ab = dist(a, b);
        double bc = dist(b, c);
        double ac = dist(a, c);
        double angle = Math.acos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc));
        return Math.toDegrees(angle);
    }

    private static double dist(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private static String classifyGesture(int fingers, MatOfPoint contour, MatOfInt4 defects) {
        if (fingers == 0) return "Fist";
        if (fingers == 5) return "Palm";

        List<Point> contourPoints = contour.toList();

        // Calcula centro da mão
        Moments m = Imgproc.moments(contour);
        double cx = m.get_m10() / m.get_m00();
        double cy = m.get_m01() / m.get_m00();

        // Pega os pontos mais altos e mais baixos
        double topY = contourPoints.stream().mapToDouble(p -> p.y).min().orElse(cy);
        double bottomY = contourPoints.stream().mapToDouble(p -> p.y).max().orElse(cy);

        // 1 dedo → verificar Thumbs Up
        if (fingers == 1) {
            Point highestPoint = contourPoints.stream().min((p1, p2) -> Double.compare(p1.y, p2.y)).get();
            if (highestPoint.y < cy) return "Thumbs Up";
            else return "Index Up"; // outro gesto de 1 dedo
        }

        // 2 dedos → Peace ou Rock
        if (fingers == 2) {
            // calcula distância horizontal entre dedos mais altos
            List<Point> topPoints = contourPoints.stream()
                    .sorted((p1, p2) -> Double.compare(p1.y, p2.y))
                    .limit(2).toList();
            double dx = Math.abs(topPoints.get(0).x - topPoints.get(1).x);
            if (dx > 40) return "Peace"; // dedos separados
            else return "Rock";           // dedos próximos
        }

        // 3 dedos → pode ser OK ou outros gestos
        if (fingers == 3) {
            // procurar círculo aproximado (polegar + indicador)
            double minDist = Double.MAX_VALUE;
            for (int i = 0; i < contourPoints.size(); i++) {
                for (int j = i + 1; j < contourPoints.size(); j++) {
                    double dist = dist(contourPoints.get(i), contourPoints.get(j));
                    if (dist < minDist) minDist = dist;
                }
            }
            if (minDist < 50) return "OK"; // aproximação de círculo
        }

        // Default para 3 ou 4 dedos
        return fingers + " Fingers";
    }

    private static void saveImage(Mat image, String filename) {
        Imgcodecs.imwrite(filename, image);
        System.out.println("Imagem salva como " + filename);
    }

    private static class FingerData {
        int count;
        double avgAngle;

        public FingerData(int count, double avgAngle) {
            this.count = count;
            this.avgAngle = avgAngle;
        }
    }
}

}
```

---

## 💡 Observações

- O código utiliza recursos do Java para monitorar uso de CPU e memória.
- O arquivo CSV é salvo em `src/src/handDetector/reports/csvs/`.
- Snapshots são salvos em `src/src/handDetector/reports/media/`.
- Para encerrar, pressione `ESC`. Para salvar imagem, pressione `2`.
- Certifique-se de que o OpenCV está corretamente instalado e configurado no seu ambiente Java.

