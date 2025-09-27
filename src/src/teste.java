import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class teste {
    public static void main(String[] args) {
        // Carrega a biblioteca nativa do OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Caminho da imagem de teste
        String imagePath = "images/reuniao.webp";
        Mat image = Imgcodecs.imread(imagePath);

        if (image.empty()) {
            System.out.println("Não foi possível carregar a imagem!");
            return;
        }

        // Carrega o classificador Haar Cascade para faces
        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println("Faces detectadas: " + faceDetections.toArray().length);

        // Desenha retângulos nas faces detectadas
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
        }

        // Salva a imagem com as faces detectadas
        Imgcodecs.imwrite("resultado.jpg", image);
        System.out.println("Imagem processada salva como resultado.jpg");
    }
}
