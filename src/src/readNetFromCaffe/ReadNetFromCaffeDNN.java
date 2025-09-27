package readNetFromCaffe;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;

public class ReadNetFromCaffeDNN {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String projectPath = "C:/SeuProjeto/DNN/readNetFromCaffe/"; // ajuste o caminho
        String protoPath = projectPath + "resources/deploy.prototxt";
        String modelPath = projectPath + "resources/res10_300x300_ssd_iter_140000.caffemodel";

        System.out.println("Carregando rede...");
        Net net = Dnn.readNetFromCaffe(protoPath, modelPath);
        System.out.println("Rede carregada com sucesso!");

        System.out.println("Carregando imagem...");
        String imagePath = projectPath + "images/input.jpg";

        // Seleção de arquivo via JFileChooser
        JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Selecionar Arquivo");
        jfc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter(
                "Image files", "jpg", "jpeg", "png"));
        int result = jfc.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {
            imagePath = jfc.getSelectedFile().getAbsolutePath();
        } else {
            System.out.println("Nenhum arquivo selecionado. Usando imagem padrão.");
        }

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("Não foi possível carregar a imagem!");
            return;
        }
        System.out.println("Imagem carregada com sucesso!");

        System.out.println("Detectando faces...");
        Mat blob = Dnn.blobFromImage(image,
                1.0,
                new Size(300, 300),
                new Scalar(104.0, 177.0, 123.0),
                false,
                false);

        net.setInput(blob);

        System.out.println("Realizando forward pass...");
        Mat detections = net.forward();

        // Reshape para 2D: [N,7] para facilitar o acesso
        Mat reshapedDetections = detections.reshape(1, (int) detections.size(2));

        for (int i = 0; i < reshapedDetections.rows(); i++) {
            double[] detection = reshapedDetections.get(i, 0);
            double confidence = detection[2];

            if (confidence > 0.5) {
                int x1 = (int) (detection[3] * image.cols());
                int y1 = (int) (detection[4] * image.rows());
                int x2 = (int) (detection[5] * image.cols());
                int y2 = (int) (detection[6] * image.rows());

                Imgproc.rectangle(image, new Point(x1, y1), new Point(x2, y2),
                        new Scalar(0, 255, 0), 2);
            }
        }

        System.out.println("Faces detectadas e retângulos desenhados!");

        // Salvar imagem de saída
        String outputPath = projectPath + "images/output/faces_dnn.jpg";
        Imgcodecs.imwrite(outputPath, image);
        System.out.println("Imagem de saída salva em: " + outputPath);
        System.out.println("Detecção finalizada!");
    }
}
