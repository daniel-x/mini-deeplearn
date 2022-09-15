package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

public class Softplus extends ActivationFunction {

	public Softplus(int inpOutSize) {
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

		for (int i = 0; i < grad_inp.length; i++) {
			grad_inp[i] = grad_out[i] / Sigmoid.calcReciprocal(inp[i]);
		}
	}

	public static float calc(float x) {
		return (float) Math.log1p(Math.exp(x));
	}

	public static void calc(float[] x, float[] y) {
		for (int i = 0; i < y.length; i++) {
			y[i] = calc(x[i]);
		}
	}

	@Override
	public String getTypeShortname() {
		return "softplus";
	}
}
