package edu.coursera.parallel;

import static edu.rice.pcdp.PCDP.async;
import static edu.rice.pcdp.PCDP.finish;

public final class Compare {
    private static double sum1;
    private static double sum2;

    Compare() {

    }

    public static void main(String[] args) {
        double[] array = new double[20000000];
        for (int i = 0; i < 20000000; i++) {
            array[i] = i + 1;
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(String.format("This is run for %d/5", i + 1));
            parallel(array);
            sequential(array);
        }
    }

    public static void sequential(final double... array) {
        sum1 = 0.;
        sum2 = 0.;
        long start = System.nanoTime();
        for (int i = 0; i < array.length / 2; i++) {
            sum1 += 1 / array[i];
        }

        for (int j = array.length / 2; j < array.length; j++) {
            sum2 += 1 / array[j];
        }
        double seqResult = sum1 + sum2;
        long seqTime = System.nanoTime() - start;
        System.out.println(String
            .format("sequential version time: %#.2f, result is %#.4f",
                seqTime / 1e6, seqResult));
    }

    public static void parallel(final double... array) {
        sum1 = 0.;
        sum2 = 0.;
        long start = System.nanoTime();
        finish(() -> {
            async(() -> {
                for (int i = 0; i < array.length / 2; i++) {
                    sum1 += 1 / array[i];
                }
            });
            for (int j = array.length / 2; j < array.length; j++) {
                sum2 += 1 / array[j];
            }
        });

        double parResult = sum1 + sum2;
        long parTime = System.nanoTime() - start;
        System.out.println(String
                .format("parallel version time: %#.2f, result is %#.4f",
                    parTime / 1e6, parResult));
    }
}
