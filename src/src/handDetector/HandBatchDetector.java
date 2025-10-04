import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.stream.*;

public class HandBatchDetector extends HandDetector{

    static String PATH = "src/src/HandDetector/benchmark/images/";

    public static void main(String[] args) throws IOException {
        String inputDir = PATH + "allimgs";
        String outputCsv = PATH + "batch_results.csv";

        try (PrintWriter csvWriter = new PrintWriter(new FileWriter(outputCsv))) {
            csvWriter.println("file,fingers,maxContourArea,centerX,centerY,convexDefects,avgAngle,gesture");

            // Lista todos os arquivos da pasta
            try (Stream<Path> paths = Files.walk(Paths.get(inputDir))) {
                paths.filter(Files::isRegularFile).forEach(path -> {

                    ContourInfo info = HandDetector.analyzeImage(path.toString(), outputCsv);
                    if (info == null) {
                        System.out.println("❌ Erro lendo ou analisando: " + path);
                        return;
                    }

                    showTextOnScreen(info.processedFrame, info.fingerData.count, info.gesture, path.toString());

                    String outputImagePath = PATH + "processed/" + path.getFileName().toString();
                    Imgcodecs.imwrite(outputImagePath, info.processedFrame);


                    csvWriter.printf(Locale.US,"%s,%d,%.2f,%.2f,%.2f,%d,%.2f,%s%n",
                            path.getFileName().toString(),
                            info.fingerData.count,
                            info.maxArea,
                            info.cx,
                            info.cy,
                            info.convexDefects,
                            info.fingerData.avgAngle,
                            info.gesture
                    );
                });
            }
        }
    }

    public static void showTextOnScreen(Mat frame, int fingerCount, String gesture,String imagePath) {
        Imgproc.putText(frame, "F" + fingerCount, new Point(5, 25),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 0, 0), 1);
        Imgproc.putText(frame, "G: " + gesture, new Point(5, 55),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 0, 0), 1);
    }

}
