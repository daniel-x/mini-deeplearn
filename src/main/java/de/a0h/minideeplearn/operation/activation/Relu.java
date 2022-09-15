package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

public class Relu extends ActivationFunction {

	public Relu(int inpOutSize) {
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
			grad_inp[i] = (inp[i] < 0.0f) ? 0.0f : grad_out[i];
		}
	}

	public static float calc(float x) {
		return Math.max(0, x);
	}

	public static void calc(float[] x, float[] y) {
		for (int i = 0; i < y.length; i++) {
			y[i] = calc(x[i]);
		}
	}

	@Override
	public String getTypeShortname() {
		return "relu";
	}
}
