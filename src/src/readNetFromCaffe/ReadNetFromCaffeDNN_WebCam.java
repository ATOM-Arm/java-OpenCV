package readNetFromCaffe;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.File;

public class ReadNetFromCaffeDNN_WebCam {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String projectPath = "C:/Users/Pedro/Documents/GitHub/JavaOpenCV/src/src/readNetFromCaffe/";
        String protoPath = projectPath + "resources/deploy.prototxt";
        String modelPath = projectPath + "resources/res10_300x300_ssd_iter_140000.caffemodel";

        // Verifica se os arquivos existem
        if (!new File(protoPath).exists() || !new File(modelPath).exists()) {
            System.out.println("Erro: Arquivo prototxt ou caffemodel não encontrado!");
            System.out.println("Proto: " + protoPath);
            System.out.println("Model: " + modelPath);
            return;
        }

        System.out.println("Carregando rede...");
        Net net = Dnn.readNetFromCaffe(protoPath, modelPath);
        System.out.println("Rede carregada com sucesso!");

        VideoCapture capture = new VideoCapture(0);
        if (!capture.isOpened()) {
            System.out.println("Não foi possível abrir a webcam!");
            return;
        }

        Mat frame = new Mat();
        long lastTime = System.nanoTime();
        double fps = 0.0;

        while (true) {
            if (!capture.read(frame) || frame.empty()) break;

            // Cria blob da imagem
            Mat blob = Dnn.blobFromImage(frame,
                    1.0,
                    new Size(300, 300),
                    new Scalar(104.0, 177.0, 123.0),
                    false,
                    false);

            net.setInput(blob);
            Mat detections = net.forward();
            Mat reshapedDetections = detections.reshape(1, (int)detections.size(2));

            int faceCount = 0;

            for (int i = 0; i < reshapedDetections.rows(); i++) {
                double[] detection = reshapedDetections.get(i, 0);
                if (detection.length != 7) continue;

                double confidence = detection[2];

                if (confidence > 0.5) { // ajuste se necessário
                    faceCount++;
                    int x1 = (int) (detection[3] * frame.cols());
                    int y1 = (int) (detection[4] * frame.rows());
                    int x2 = (int) (detection[5] * frame.cols());
                    int y2 = (int) (detection[6] * frame.rows());

                    Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2),
                            new Scalar(0, 255, 0), 2);
                    Imgproc.putText(frame, "Face " + faceCount, new Point(x1, y1 - 10),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 2);
                }
            }

            // FPS
            long currentTime = System.nanoTime();
            fps = 1e9 / (currentTime - lastTime);
            lastTime = currentTime;

            // Informações na tela
            Imgproc.putText(frame, "Total Faces: " + faceCount, new Point(10, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);
            Imgproc.putText(frame, String.format("FPS: %.1f", fps), new Point(10, 60),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 0, 0), 2);

            // Print de depuração se não houver faces detectadas
            if (faceCount == 0) {
                System.out.println("Nenhuma face detectada neste frame.");
            }

            HighGui.imshow("WEBCAM - OpenCV DNN", frame);

            if (HighGui.waitKey(1) == 27) break; // ESC para sair
        }

        capture.release();
        HighGui.destroyAllWindows();
    }
}
