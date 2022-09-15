package de.a0h.minideeplearn.operation.loss;

import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.minideeplearn.operation.activation.ActivationFunction;
import de.a0h.minideeplearn.operation.gradient.VectorGradient;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

/**
 * A combination of an activation function and a loss function.
 */
public abstract class CombinedLossFunction extends ActivationFunction implements LossFunction {

	public float loss;

	public CombinedLossFunction(int inpOutSize) {
		super(inpOutSize);
	}

	@Override
	public boolean hasLoss() {
		return true;
	}

	@Override
	public float getLoss() {
		return loss;
	}

	/**
	 * Based on the loss function, this method calculates the gradient in respect to
	 * the output of the activation function part of this combined loss function.
	 * Usually, this isn't needed when training or using a neural network. However
	 * it's interesting for insights into how the network works and for other
	 * applications.
	 */
	public abstract void calcOutputGradient(float[] inp, float[] target, VectorGradient grad);

	@Override
	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {
		super.toStringBuilderWithLayoutAndValues(buf, indent);
		buf.append("\n");
		String indentSpaces = StringUtil.getSpaces(indent);

		buf.append(indentSpaces).append("    loss: ");

		NumberStats numberStats = new NumberStats();
		numberStats.aggregate(loss);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		MnFormat.toStringBuilder(loss, buf, format);

		return buf;
	}
}