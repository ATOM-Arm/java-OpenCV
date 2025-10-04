import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class handDetector {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

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
        try (PrintWriter csvWriter = new PrintWriter(new FileWriter("performance.csv"))) {
            csvWriter.println("timestamp_ms,dedos_detectados");

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

                fingers = countFingers(defects, approxContour);
                gesture = classifyGesture(fingers);

                Imgproc.putText(frame,
                        "Dedos: " + fingers + " - " + gesture,
                        new Point(20, 40),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        1.0, new Scalar(0, 255, 0), 2);
            }

            // Salva dados no CSV
            long timestamp = System.currentTimeMillis() - startTime;
            csvWriter.printf("%d,%d%n", timestamp, fingers);
            csvWriter.flush();

            // Mostra apenas uma janela com o resultado final
            HighGui.imshow("Detecção de Mão", frame);

            int key = HighGui.waitKey(100) & 0xFF; // aumenta o tempo de espera

            if (key == 27) {
                break;
            } else if (key == 50) {
                String filename = String.format(
                        "src/src/handDetector/reports/001/media/hand_snapshot_%03d.png",
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

    private static int countFingers(MatOfInt4 defects, MatOfPoint contour) {
        if (defects.empty()) return 0;
        int[] arr = defects.toArray();
        Point[] points = contour.toArray();
        int count = 0;
        for (int i = 0; i < arr.length; i += 4) {
            int startIdx = arr[i];
            int endIdx = arr[i + 1];
            int farIdx = arr[i + 2];
            float depth = arr[i + 3] / 256.0f;
            if (depth > 25) { // aumenta profundidade mínima para melhor precisão
                double angle = calcAngle(points[startIdx], points[farIdx], points[endIdx]);
                if (angle < 85) count++; // ângulo máximo ajustado
            }
        }
        return Math.min(5, count + 1);
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

    private static String classifyGesture(int fingers) {
        if (fingers == 0) return "Fist";
        if (fingers == 5) return "Palm";

        switch (fingers) {
            case 0:
                return "Fist";
            case 1:
                return "1 Fingers";
            case 2:
                return "2 Fingers";
            case 3:
                return "3 Fingers";
            case 4:
                return "4 Fingers";
            case 5:
                return "5 Fingers";
            default:
                return "Nada encontrado";
        }
    }

    private static void saveImage(Mat image, String filename) {
        Imgcodecs.imwrite(filename, image);
        System.out.println("Imagem salva como " + filename);
    }
}
