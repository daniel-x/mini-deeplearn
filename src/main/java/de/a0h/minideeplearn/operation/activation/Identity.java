package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;
import de.a0h.mininum.MnLinalg;

public class Identity extends ActivationFunction {

	public Identity(int inpOutSize) {
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

		MnLinalg.assign(grad_out, grad_inp);
	}

	public static float calc(float x) {
		return x;
	}

	public static void calc(float[] x, float[] y) {
		for (int i = 0; i < y.length; i++) {
			y[i] = calc(x[i]);
		}
	}

	@Override
	public String getTypeShortname() {
		return "id";
	}
}
