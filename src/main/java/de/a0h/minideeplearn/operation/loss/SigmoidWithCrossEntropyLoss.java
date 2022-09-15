package de.a0h.minideeplearn.operation.loss;

import de.a0h.minideeplearn.operation.activation.Sigmoid;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;

/**
 * A combination of the 1-dimensional sigmoid function and cross entropy loss.
 * Combining these two makes great sense because in consequence the maths of the
 * gradient calculation falls into place nicely, which makes it simple and fast
 * to calculate gradients in this class.
 */
public class SigmoidWithCrossEntropyLoss extends CombinedLossFunction {

	public SigmoidWithCrossEntropyLoss(int inpOutSize) {
		super(inpOutSize);
	}

	@Override
	public float[] calcOutput(float[] inp) {
		Sigmoid.calc(inp, out);

		return out;
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		loss = CrossEntropyLoss.crossEntropy_zeroSafe(out, target);
		return loss;
	}

	@Override
	public void calcOutputGradient(float[] inp, float[] target, VectorGradient grad) {
		float[] grad_out = grad.inp;
		CrossEntropyLoss.crossEntropyGradient_zeroSafe(out, target, grad_out);
	}

	@Override
	public void calcGradient(float[] inp, float[] target, Gradient grad) {
		VectorGradient grad_ = (VectorGradient) grad;

		float[] grad_inp = grad_.inp;

		if (inp.length == 1) {
			// twofold classification (aka binary classification)
			grad_inp[0] = out[0] - target[0];

		} else {
			// manifold classification (aka multinomial classification)
			float nReciprocal = 1.0f / inp.length;

			for (int i = 0; i < out.length; i++) {
				grad_inp[i] = nReciprocal * target[i] * (out[i] - 1.0f);
			}
		}
	}

	@Override
	public String getTypeShortname() {
		return "sigmoid-ce";
	}
}
