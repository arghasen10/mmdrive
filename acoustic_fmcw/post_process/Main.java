import mat4j.FFT2D;
import org.apache.commons.math3.complex.Complex;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        RecordingParser fileParser = new RecordingParser("resources/RawDataCollection.txt");
        fileParser.parseRecording();
        SignalProcessor signalProcessor = new SignalProcessor();
        ArrayList<Data> dataNodes = fileParser.getDataNodes();

        int i = 0;
        ArrayList<Complex[][]> rangeDoppler = new ArrayList<>();
        while (i < dataNodes.size()-1) {
            int counter = 0;
            Complex[][] x = new Complex[13][];
            while(counter < 13) {
                System.out.println("Processing index: " + i);

                Data data = dataNodes.get(i);
                double[] chirp = signalProcessor.toArrayManual(data.getChirp());
                double[] direct = signalProcessor.toArrayManual(data.getDirect());
                double[] record = signalProcessor.toArrayManual(data.getRecord());

                x[counter] = signalProcessor.FourierTransform(chirp, direct, record);
                i++;
                counter += 1;
            }

            System.out.println("Calculating 2DFFT");
            FFT2D fft2D = new FFT2D();
            Complex[][] y = FFT2D.fft(x);
            rangeDoppler.add(y);
        }

    }
}