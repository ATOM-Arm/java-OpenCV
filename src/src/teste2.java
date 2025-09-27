import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class teste2 {
    public static void main(String[] args) {
        // Carrega a biblioteca nativa do OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Inicia captura de vídeo da webcam (0 = câmera padrão)
        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Não foi possível abrir a webcam!");
            return;
        }

        // Carrega o classificador Haar Cascade para faces
        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");

        Mat frame = new Mat();
        while (true) {
            if (!camera.read(frame) || frame.empty()) {
                System.out.println("Não foi possível capturar o frame.");
                break;
            }

            // Detecta faces
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            // Desenha retângulos verdes nas faces detectadas
            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            }

            // Mostra o frame
            HighGui.imshow("Detecção de Faces - Pressione ESC para sair", frame);

            // Sai do loop se a tecla ESC for pressionada
            if (HighGui.waitKey(1) == 27) {
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
