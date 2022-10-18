package mat4j;


/*
 * Copyright (c) 2011, 2020, Frank Jiang and/or its affiliates. All rights
 * reserved.
 * FFT2D.java is PROPRIETARY/CONFIDENTIAL built in 2013.
 * Use is subject to license terms.
 */

import org.apache.commons.math3.complex.Complex;

/**
 * The implementation of two dimensional Fast Fourier Transformation.
 * <p>
 * This is the interface of recalling the Fourier Transformation part of Open
 * Source Physics Project.
 * </p>
 *
 * @author <a href="mailto:jiangfan0576@gmail.com">Frank Jiang</a>
 * @version 1.0.0
 */
public class FFT2D
{
    /**
     * Perform the 2D Fast Fourier Transformation.
     *
     * @param x
     *            the specified complex matrix, <code>x</code> must arranged as
     *            <code>x[rows][columns]</code>
     * @return the result of 2D Fast Fourier Transformation
     */
    public static Complex[][] fft(Complex[][] x)
    {
        int nrows = x.length;
        if (nrows < 1 || x[0].length < 1)
            throw new IllegalArgumentException(Messages.getString("FFT2D.InvalidSize")); //$NON-NLS-1$
        int ncols = x[0].length;
        double[] data = new double[nrows * ncols * 2];
        for (int j = 0; j < nrows; j++)
            for (int i = 0; i < ncols; i++)
            {
                data[j * ncols * 2 + 2 * i] = x[j][i].getReal();
                data[j * ncols * 2 + 2 * i + 1] = x[j][i].getImaginary();
            }
        new DoubleFFT_2D(nrows, ncols).complexForward(data);
        Complex[][] y = new Complex[nrows][ncols];
        for (int j = 0; j < nrows; j++)
            for (int i = 0; i < ncols; i++)
                y[j][i] = new Complex(data[j * ncols * 2 + 2 * i], data[j * ncols * 2
                        + 2 * i + 1]);
        return y;
    }

    /**
     * Perform the 2D Inverse Fast Fourier Transformation.
     *
     * @param y
     *            the specified complex matrix, <code>y</code> must arranged as
     *            <code>y[rows][columns]</code>
     * @param scale
     *            <tt>true</tt> if scaling is need to be performed
     * @return the result of 2D Inverse Fast Fourier Transformation
     */
    public static Complex[][] ifft(Complex[][] y, boolean scale)
    {
        int nrows = y.length;
        if (nrows < 1 || y[0].length < 1)
            throw new IllegalArgumentException(Messages.getString("FFT2D.InvalidSize")); //$NON-NLS-1$
        int ncols = y[0].length;
        int n = nrows * ncols;
        double[] data = new double[n * 2];
        for (int j = 0; j < nrows; j++)
            for (int i = 0; i < ncols; i++)
            {
                data[j * ncols * 2 + 2 * i] = y[j][i].getReal();
                data[j * ncols * 2 + 2 * i + 1] = y[j][i].getImaginary();
            }
        new DoubleFFT_2D(nrows, ncols).complexInverse(data, scale);
        Complex[][] x = new Complex[nrows][ncols];
        for (int j = 0; j < nrows; j++)
            for (int i = 0; i < ncols; i++)
                x[j][i] = new Complex(data[j * ncols * 2 + 2 * i], data[j * ncols * 2
                        + 2 * i + 1]);
        return x;
    }

    /**
     * Perform the 2D Inverse Fast Fourier Transformation.
     *
     * @param y
     *            the specified complex matrix, <code>y</code> must arranged as
     *            <code>y[rows][columns]</code>
     * @return the result of 2D Inverse Fast Fourier Transformation
     */
    public static Complex[][] ifft(Complex[][] y)
    {
        return ifft(y, true);
    }
}