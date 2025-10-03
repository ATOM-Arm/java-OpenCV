package readNetFromCaffe;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

public class ObjectDNN {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Carregar o modelo MobileNet-SSD
        String protoFile = "resources/MobileNetSSD_deploy.prototxt";
        String weightsFile = "resources/MobileNetSSD_deploy.caffemodel";

        Net net = Dnn.readNetFromCaffe(protoFile, weightsFile);

        // Lista de classes suportadas
        String[] classNames = {
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        // Carregar imagem
        Mat image = Imgcodecs.imread("resources/transito.webp");
        if (image.empty()) {
            System.out.println("Erro ao carregar imagem!");
            return;
        }

        // Criar blob para passar à rede
        Mat blob = Dnn.blobFromImage(image, 0.007843, new Size(300, 300),
                new Scalar(127.5, 127.5, 127.5), false, false);

        net.setInput(blob);

        // Forward pass
        Mat detections = net.forward();

        // Processar detecções
        detections = detections.reshape(1, (int) detections.total() / 7);
        for (int i = 0; i < detections.rows(); i++) {
            double confidence = detections.get(i, 2)[0];

            if (confidence > 0.4) { // limiar de confiança
                int classId = (int) detections.get(i, 1)[0];

                int left = (int) (detections.get(i, 3)[0] * image.cols());
                int top = (int) (detections.get(i, 4)[0] * image.rows());
                int right = (int) (detections.get(i, 5)[0] * image.cols());
                int bottom = (int) (detections.get(i, 6)[0] * image.rows());

                // Desenhar retângulo
                Imgproc.rectangle(image, new Point(left, top), new Point(right, bottom),
                        new Scalar(0, 255, 0), 2);

                // Nome da classe
                String label = classNames[classId] + " (" + String.format("%.2f", confidence) + ")";
                Imgproc.putText(image, label, new Point(left, top - 5),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
            }
        }

        // Mostrar resultado
        HighGui.imshow("Reconhecimento de Objetos - MobileNetSSD", image);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
    }
}
