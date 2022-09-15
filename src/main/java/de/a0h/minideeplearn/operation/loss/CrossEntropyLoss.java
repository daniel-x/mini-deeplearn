package de.a0h.minideeplearn.operation.loss;

import java.util.Random;

import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

public class CrossEntropyLoss implements LossFunction {

	public float loss;

	public int inpSize;

	public CrossEntropyLoss(int inpSize, int outSizeIgnored) {
		this.inpSize = inpSize;
	}

	@Override
	public int getInputSize() {
		return inpSize;
	}

	@Override
	public boolean hasOutput() {
		return false;
	}

	@Override
	public int getOutputSize() {
		OperationUtil.throwOutputNotProvidedException(getClass());
		return -1;
	}

	@Override
	public float[] calcOutput(float[] inp) {
		OperationUtil.throwOutputNotProvidedException(getClass());
		return null;
	}

	@Override
	public float[] getOutput() {
		OperationUtil.throwOutputNotProvidedException(getClass());
		return null;
	}

	@Override
	public boolean hasLoss() {
		return true;
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		loss = crossEntropy_zeroSafe(inp, target);

		return loss;
	}

	@Override
	public float getLoss() {
		return loss;
	}

	@Override
	public void calcGradient(float[] inp, float[] target, Gradient grad) {
		VectorGradient grad_ = (VectorGradient) grad;

		crossEntropyGradient_zeroSafe(inp, target, grad_.inp);
	}

	@Override
	public VectorGradient createGradient() {
		return new VectorGradient(inpSize);
	}

	/**
	 * Returns immediately.
	 */
	@Override
	public void initParams(Random rnd) {
	}

	/**
	 * Returns immediately.
	 */
	@Override
	public void learn(Gradient grad, float negLearningRate) {
	}

	public static final float crossEntropy_zeroSafe(float[] predicted, float[] target) {
		float result;

		if (predicted.length == 1) {
			if (target[0] == 1.0f) {
				result = -log_zeroSafe(predicted[0]);
			} else if (target[0] == 0.0f) {
				result = -log_zeroSafe(1.0f - predicted[0]);
			} else {
				result = -target[0] * log_zeroSafe(predicted[0]) //
						- (1.0f - target[0]) * log_zeroSafe(1.0f - predicted[0]);
			}

		} else {
			result = 0.0f;

			for (int i = 0; i < predicted.length; i++) {
				if (target[i] == 0.0f) {
					continue;
				}

				result -= target[i] * log_zeroSafe(predicted[i]);
			}

			result /= predicted.length;
		}

		return result;
	}

	public static void crossEntropyGradient_zeroSafe(float[] predicted, float[] target, float[] grad_inp) {
		if (predicted.length == 1) {
			// twofold classification (binary classification)
			grad_inp[0] = (1.0f - target[0]) / (1.0f - predicted[0]) - target[0] / predicted[0];

		} else {
			// manifold classification (multinomial classification)
			float negNReciprocal = -1.0f / predicted.length;

			for (int i = 0; i < predicted.length; i++) {
				grad_inp[i] = negNReciprocal * (target[i] / predicted[i]);
			}
		}
	}

	public static final float log_zeroSafe(float x) {
		return log(x + Float.MIN_NORMAL);
	}

	public static final float log(float x) {
		return (float) Math.log(x);
	}

	@Override
	public String toString() {
		return toStringWithLayout();
	}

	@Override
	public String getTypeShortname() {
		return "ce";
	}

	@Override
	public String toStringWithLayout() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayout(buf);
		return buf.toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		buf.append(getTypeShortname());
		return buf;
	}

	@Override
	public String toStringWithLayoutAndValues() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayoutAndValues(buf, 0);
		return buf.toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {
		String indentSpaces = StringUtil.getSpaces(indent);
		buf.append(indentSpaces);

		toStringBuilderWithLayout(buf).append(": ").append("loss:");

		NumberStats numberStats = new NumberStats();
		numberStats.aggregate(loss);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		MnFormat.toStringBuilder(loss, buf, format);

		return buf;
	}
}
