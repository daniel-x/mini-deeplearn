package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

/**
 * Swish activation function (paper: Swish: a self-gated activation function,
 * Google, 2017) for scalars and vectors.
 */
public class Swish extends ActivationFunction {

	protected float[] sig;

	public Swish(int inpOutSize) {
		super(inpOutSize);
		sig = new float[inpOutSize];
	}

	@Override
	public float[] calcOutput(float[] inp) {
		calc(inp, sig, out);

		return out;
	}

	@Override
	public void calcGradient(float[] inp, float[] grad_out, Gradient grad) {
		VectorGradient grad_ = (VectorGradient) grad;

		float[] grad_inp = grad_.inp;

		for (int i = 0; i < out.length; i++) {
			grad_inp[i] = (out[i] + sig[i] * (1.0f - out[i])) * grad_out[i];
		}
	}

	public static void calc(float[] x, float[] sig, float[] y) {
		for (int i = 0; i < y.length; i++) {
			sig[i] = 1.0f / (1.0f + (float) Math.exp(-x[i]));
			y[i] = x[i] * sig[i];
		}
	}

	@Override
	public String getTypeShortname() {
		return "swish";
	}
}
