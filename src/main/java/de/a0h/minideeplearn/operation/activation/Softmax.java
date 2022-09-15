package de.a0h.minideeplearn.operation.activation;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.mininum.MnFuncs;
import de.a0h.mininum.MnLinalg;

public class Softmax extends ActivationFunction {

	public Softmax(int inpOutSize) {
		super(inpOutSize);
	}

	@Override
	public float[] calcOutput(float[] inp) {
		calc(inp, out);

		return out;
	}

	@Override
	public void calcGradient(float[] inp, float[] grad_out, Gradient grad) {
		if (Math.random() <= 1.0f) {
			throw new UnsupportedOperationException(
					"sole softmax derivative is not supported. Use a combined softmax instead, e.g. SoftmaxWithCrossEntropyLoss.");
		}
	}

	public static float calc(float x) {
		throw new UnsupportedOperationException("not applicable for 1 element");
	}

	public static void calc(float[] x, float[] y) {
		float max = MnFuncs.max(x);

		float sum = 0;
		for (int i = 0; i < y.length; i++) {
			y[i] = (float) Math.exp(x[i] - max);
			sum += y[i];
		}

		MnLinalg.div(y, sum);
	}

	@Override
	public String getTypeShortname() {
		return "softmax";
	}
}
