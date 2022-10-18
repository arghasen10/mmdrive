import com.github.psambit9791.jdsp.signal.CrossCorrelation;
import com.github.psambit9791.jdsp.transform.DiscreteFourier;
import com.github.psambit9791.jdsp.transform.Hilbert;
import org.apache.commons.math3.complex.Complex;
import java.util.ArrayList;

public class SignalProcessor {
    private int SAMPLE_LENGTH = 2048;
    private int NUMBER_OF_INITIAL = 5;
    private int NUMBER_OF_SAMPLE_POINTS = 15;
    private double[][] mixedSamplesPhase;
    private double[][] initialPoints;
    private double[] distances;
    private double[] center;


    public SignalProcessor() {
        mixedSamplesPhase = new double[NUMBER_OF_INITIAL][SAMPLE_LENGTH];
        initialPoints = new double[NUMBER_OF_SAMPLE_POINTS][2];
        distances = new double[NUMBER_OF_SAMPLE_POINTS];
        center = new double[]{0.0, 0.0};
    }
    double[] signalMultiplication(double[] x, double[] y){
        CrossCorrelation cc = new CrossCorrelation(x, y);
        double[] out = cc.crossCorrelate("valid");

        int maxAt = 0;
        for(int i=0; i<out.length; i++){
            maxAt = out[i] > out[maxAt] ? i : maxAt;
        }

        double[] sample = new double[x.length - maxAt];
        for(int i=maxAt; i<x.length; i++){
            sample[i-maxAt] = x[i];
        }
        return sample;
    }
    Complex[] FourierTransform(double[] chirp, double[] direct, double[] record)
    {
        double[] rSample = signalMultiplication(chirp, record);
        double[] dSample = signalMultiplication(chirp, direct);

        // CHANGES
        // hilbertTransform renamed to transform

        Hilbert h = new Hilbert(rSample);
        h.transform();
        double[][] analyticRSample = h.getOutput();

        Hilbert d = new Hilbert(dSample);
        d.transform();
        double[][] analyticDSample = d.getOutput();

        Hilbert f = new Hilbert(direct);
        f.transform();
        double[][] analyticDirect = f.getOutput();

        Hilbert g = new Hilbert(record);
        g.transform();
        double[][] analyticRecord = g.getOutput();

        double[] mixedRSignal = new double[SAMPLE_LENGTH];
        double[] mixedDSignal = new double[SAMPLE_LENGTH];

        for(int i=0; i<SAMPLE_LENGTH; i++)
        {
            mixedRSignal[i] = analyticRecord[i][0]*analyticRSample[i][0] + analyticRecord[i][1]*analyticRSample[i][1];
            mixedDSignal[i] = analyticDirect[i][0]*analyticDSample[i][0] + analyticDirect[i][1]*analyticDSample[i][1];
        }

        DiscreteFourier fft = new DiscreteFourier(mixedRSignal);
        fft.transform();
        Complex[] mixedRFftComplex = fft.getComplex(true);

        DiscreteFourier fftD = new DiscreteFourier(mixedDSignal);
        fftD.transform();
        Complex[] mixedDFftcomplex = fftD.getComplex(true);

        long time = System.currentTimeMillis();

        double[] amplitude = new double[mixedRFftComplex.length];
        double[] angle = new double[mixedRFftComplex.length];
        Complex[] ifSignal = new Complex[mixedRFftComplex.length];
        for(int i=0; i<mixedRFftComplex.length; i++){

            Complex value = mixedRFftComplex[i].subtract(mixedDFftcomplex[i]);
            amplitude[i] = value.abs();
            angle[i] = value.getArgument();
            ifSignal[i] = value;
        }
        return ifSignal;
    }

    public double[] toArrayManual(ArrayList<Double> arrayList) {
        double[] result = new double[arrayList.size()];
        for(int i = 0; i < arrayList.size(); i++) {
            result[i] = arrayList.get(i);
        }
        return result;
    }

    public double getMean(double[] arr) {
        double sum = 0.0;
        for(int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum / arr.length;
    }

}
