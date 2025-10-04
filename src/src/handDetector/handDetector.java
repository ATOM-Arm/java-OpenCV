
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

public class HandDetector {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    static String PATH = "src/src/HandDetector/reports/001/";


    public static void main(String[] args) throws IOException {

        VideoCapture camera = createCamera(0);

        Mat frame = new Mat();
        Mat mask = new Mat();
        Mat hierarchy = new Mat();

        try (PrintWriter csvWriter = new PrintWriter(new FileWriter(PATH + "csvs/performance" + (int) (Math.random() * 1000) + ".csv"))) {
            csvWriter.println("frame,fingers,maxContourArea,centerX,centerY,convexDefects,avgAngle,fps,usedMemoryMB,cpuLoad,gesture");

            int frameNumber = 0;

            while (true) {
                long startTime = System.currentTimeMillis();
                if (!camera.read(frame) || frame.empty()) break;

                processFrame(frame, mask, hierarchy);

                int index = findLargestContourIndex(mask, hierarchy);
                FingerData fingerData = new FingerData(0, 0);
                String gesture = "";
                double maxArea = 0, cx = 0, cy = 0;
                int convexDefects = 0;

                if (index != -1) {
                    ContourInfo info = analyzeHand(frame, mask, hierarchy, index);
                    fingerData = info.fingerData;
                    gesture = info.gesture;
                    maxArea = info.maxArea;
                    cx = info.cx;
                    cy = info.cy;
                    convexDefects = info.convexDefects;
                }

                double fps = calculateFPS(startTime);
                double usedMemoryMB = getUsedMemoryMB();
                double cpuLoad = getCpuLoad();

                if (index != -1) {
                    csvWriter.printf(Locale.US, "%d,%d,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.4f,%s%n",
                            frameNumber++, fingerData.count, maxArea, cx, cy, convexDefects,
                            fingerData.avgAngle, fps, usedMemoryMB, cpuLoad, gesture);
                    csvWriter.flush();
                    showTextOnScreen(frame, fingerData, gesture, fps, usedMemoryMB, cpuLoad);
                }

                // Mostra apenas uma janela com o resultado final
                HighGui.imshow("Detecção de Mão", frame);

                int key = HighGui.waitKey(100) & 0xFF; // aumenta o tempo de espera

                if (key == 27) {
                    break;
                } else if (key == 50) {
                    String filename = String.format(
                            PATH + "media/hand_snapshot_%03d.png",
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

    public static VideoCapture createCamera(int index) {
        VideoCapture camera = new VideoCapture(index);
        if (!camera.isOpened()) {
            System.out.println("❌ Erro ao abrir a webcam.");
        }
//        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
//        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);
        return camera;
    }

    public static void showTextOnScreen(Mat frame, FingerData fingerData, String gesture, double fps, double usedMemoryMB, double cpuLoad) {
        // Mostra na tela
        Imgproc.putText(
                frame,
                "Dedos: " + fingerData.count + " - " + gesture,
                new Point(20, 40),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0, new Scalar(0, 255, 0), 2);

        Imgproc.putText(
                frame,
                String.format("FPS: %.2f CPU: %.2f%% Mem: %.2f)", fps, cpuLoad * 100, usedMemoryMB),
                new Point(20, 80),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7, new Scalar(0, 0, 0), 4);
        Imgproc.putText(
                frame,
                String.format("FPS: %.2f CPU: %.2f%% Mem: %.2f)", fps, cpuLoad * 100, usedMemoryMB),
                new Point(20, 80),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7, new Scalar(255, 50, 50), 2);
    }

    public static void showTextOnScreen(Mat frame, int fingers, String gesture) {
        Imgproc.putText(
                frame,
                "Dedos: " + fingers + " - " + gesture,
                new Point(20, 40),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0, new Scalar(0, 255, 0), 2);
    }

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

    private static void processFrame(Mat frame, Mat mask, Mat hierarchy) {
        Core.flip(frame, frame, 1);
        Imgproc.GaussianBlur(frame, frame, new Size(5, 5), 0);
        Mat ycrcb = new Mat();
        Imgproc.cvtColor(frame, ycrcb, Imgproc.COLOR_BGR2YCrCb);
        Core.inRange(ycrcb, new Scalar(0, 133, 77), new Scalar(255, 173, 127), mask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.medianBlur(mask, mask, 5);
    }

    private static int findLargestContourIndex(Mat mask, Mat hierarchy) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea = 5000;
        int index = -1;
        for (int i = 0; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                index = i;
            }
        }
        return index;
    }

    public static ContourInfo analyzeHand(Mat frame, Mat mask, Mat hierarchy, int index) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint contour = contours.get(index);
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        Imgproc.approxPolyDP(contour2f, contour2f, 3, true);
        MatOfPoint approxContour = new MatOfPoint();
        contour2f.convertTo(approxContour, CvType.CV_32S);
        Imgproc.drawContours(frame, contours, index, new Scalar(0, 255, 0), 2);
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(approxContour, hull);
        MatOfPoint hullPoints = hullPointsFromIndices(approxContour, hull);
        List<MatOfPoint> hullList = new ArrayList<>();
        hullList.add(hullPoints);
        Imgproc.drawContours(frame, hullList, 0, new Scalar(255, 0, 0), 2);
        MatOfInt4 defects = new MatOfInt4();
        Imgproc.convexityDefects(approxContour, hull, defects);
        FingerData fingerData = countFingers(defects, approxContour);
        String gesture = classifyGesture(fingerData.count, approxContour, defects);
        Moments m = Imgproc.moments(contour);
        double cx = m.get_m10() / m.get_m00();
        double cy = m.get_m01() / m.get_m00();
        double maxArea = Imgproc.contourArea(contour);
        int convexDefects = (int) defects.total();
        return new ContourInfo(fingerData, gesture, maxArea, cx, cy, convexDefects, frame);
    }

    private static double calculateFPS(long startTime) {
        long endTime = System.currentTimeMillis();
        return 1000.0 / (endTime - startTime);
    }

    private static double getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / 1024.0 / 1024.0;
    }

    private static double getCpuLoad() {
        com.sun.management.OperatingSystemMXBean osBean =
                (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        return osBean.getProcessCpuLoad();
    }

    private static void saveSnapshot(Mat frame) {
        String filename = String.format(PATH + "media/hand_snapshot_%03d.png", (int) (Math.random() * 1000));
        saveImage(frame, filename);
        System.out.println("📸 Imagem salva como " + filename);
    }

    public static ContourInfo analyzeImage(String imagePath, String csvPath) {
        Mat frame = Imgcodecs.imread(imagePath);
        if (frame.empty()) {
            System.out.println("❌ Erro ao carregar imagem: " + imagePath);
            return null;
        }

        Mat mask = new Mat();
        Mat hierarchy = new Mat();

        // pré-processamento
        processFrame(frame, mask, hierarchy);

        int index = findLargestContourIndex(mask, hierarchy);
        if (index == -1) {
            System.out.println("Nenhuma mão detectada em: " + imagePath);
            return null;
        }

        // análise
        ContourInfo info = analyzeHand(frame, mask, hierarchy, index);

        return info;
    }



}
