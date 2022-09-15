package de.a0h.minideeplearn.operation.activation;

import java.util.Random;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

/**
 * Activation functions should implement this class.
 */
public abstract class ActivationFunction implements Operation {

	public float[] out;

	public ActivationFunction(int inpOutSize) {
		out = new float[inpOutSize];
	}

	@Override
	public int getInputSize() {
		return out.length;
	}

	@Override
	public boolean hasOutput() {
		return true;
	}

	@Override
	public int getOutputSize() {
		return getInputSize();
	}

	@Override
	public float[] getOutput() {
		return out;
	}

	@Override
	public VectorGradient createGradient() {
		return new VectorGradient(getInputSize());
	}

	@Override
	public boolean hasLoss() {
		return false;
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		OperationUtil.throwLossNotProvidedException(getClass());
		return -1.0f;
	}

	@Override
	public float getLoss() {
		OperationUtil.throwLossNotProvidedException(getClass());
		return -1.0f;
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
	public void learn(Gradient grad, float learningRate) {
	}

	@Override
	public String toString() {
		return toStringWithLayout();
	}

	@Override
	public String toStringWithLayout() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayout(buf);
		return buf.toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		buf.append(getTypeShortname()).append("[").append(getInputSize()).append("]");
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
		toStringBuilderWithLayout(buf).append(": out: ");

		NumberStats numberStats = new NumberStats();
		numberStats.aggregate(out);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		MnFormat.toStringBuilder(out, buf, format);

		return buf;
	}
}