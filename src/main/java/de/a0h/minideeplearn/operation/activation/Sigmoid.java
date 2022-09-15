package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

public class Sigmoid extends ActivationFunction {

	public Sigmoid(int inpOutSize) {
		super(inpOutSize);
	}

	@Override
	public float[] calcOutput(float[] inp) {
		calc(inp, out);

		return out;
	}

	@Override
	public void calcGradient(float[] inp, float[] grad_out, Gradient grad) {
		VectorGradient grad_ = (VectorGradient) grad;

		float[] grad_inp = grad_.inp;

		for (int i = 0; i < out.length; i++) {
			grad_inp[i] = (out[i] * (1.0f - out[i])) * grad_out[i];
		}
	}

	public static float calc(float x) {
		return 1.0f / (1.0f + (float) Math.exp(-x));
	}

	public static float calcReciprocal(float x) {
		return (1.0f + (float) Math.exp(-x));
	}

	public static void calc(float[] x, float[] y) {
		for (int i = 0; i < y.length; i++) {
			y[i] = calc(x[i]);
		}
	}

	@Override
	public String getTypeShortname() {
		return "sigmoid";
	}
}
