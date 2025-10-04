import org.opencv.core.Mat;

public class ContourInfo {
    FingerData fingerData;
    String gesture;
    double maxArea, cx, cy;
    int convexDefects;
    Mat processedFrame;

    ContourInfo(FingerData fd, String g, double a, double x, double y, int d, Mat frame) {
        fingerData = fd;
        gesture = g;
        maxArea = a;
        cx = x;
        cy = y;
        convexDefects = d;
        processedFrame = frame;
    }
}