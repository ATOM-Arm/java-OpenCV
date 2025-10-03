import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class Object {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Carrega o classificador pré-treinado (exemplo: carros, disponível no GitHub do OpenCV)
        CascadeClassifier carCascade = new CascadeClassifier("resources/cars.xml");

        // Carregar imagem de teste
        Mat image = Imgcodecs.imread("resources/street.jpg");

        // Detectar objetos
        MatOfRect cars = new MatOfRect();
        carCascade.detectMultiScale(image, cars);

        // Desenhar retângulos
        for (Rect rect : cars.toArray()) {
            Imgproc.rectangle(image, rect, new Scalar(0, 255, 0), 2);
        }

        // Salvar imagem processada
        Imgcodecs.imwrite("cars_detected.jpg", image);

        System.out.println("Reconhecimento de objetos concluído!");
    }
}
