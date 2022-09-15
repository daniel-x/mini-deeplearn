package de.a0h.minideeplearn.operation.loss;

import de.a0h.minideeplearn.operation.activation.Softmax;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

/**
 * A combination of the softmax function and cross entropy loss. Combining these
 * two makes great sense because in consequence the maths of the gradient
 * calculation falls into place nicely, which makes it simple and fast to
 * calculate gradients in this class.
 */
public class SoftmaxWithCrossEntropyLoss extends CombinedLossFunction {

	public SoftmaxWithCrossEntropyLoss(int inpOutSize) {
		super(inpOutSize);
	}

	@Override
	public float[] calcOutput(float[] inp) {
		Softmax.calc(inp, out);

		return out;
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		loss = CrossEntropyLoss.crossEntropy_zeroSafe(out, target);
		return loss;
	}

	@Override
	public void calcGradient(float[] inp, float[] target, Gradient grad) {
		VectorGradient grad_ = (VectorGradient) grad;

		float[] grad_inp = grad_.inp;

		float nReciprocal = 1.0f / inp.length;

		for (int i = 0; i < out.length; i++) {
			grad_inp[i] = nReciprocal * (out[i] - target[i]);
		}
	}

	@Override
	public void calcOutputGradient(float[] inp, float[] target, VectorGradient grad) {
		throw new UnsupportedOperationException("" + //
				"the sole softmax derivative is not supported, " + //
				"because it is inefficient and usually not used");
	}

	@Override
	public String getTypeShortname() {
		return "softmax-ce";
	}
}
