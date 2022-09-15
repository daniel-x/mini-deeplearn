package de.a0h.minideeplearn.operation.layer;

import java.util.Random;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.minideeplearn.operation.gradient.DenseGradient;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.mininum.MnFuncs;
import de.a0h.mininum.MnLinalg;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

/**
 * Dense layer of the form y = w * x + b, where</br>
 * x: input vector</br>
 * y: output vector</br>
 * w: weights matrix (learnable parameters)</br>
 * b: bias (learnable parameters)</br>
 */
public class Dense implements Operation {

	public float[] out;

	public float[][] weights;

	public float[] bias;

	public Dense(int inpSize, int outSize) {
		out = new float[outSize];
		weights = new float[outSize][inpSize];
		bias = new float[outSize];
	}

	@Override
	public int getInputSize() {
		return weights[0].length;
	}

	@Override
	public boolean hasOutput() {
		return true;
	}

	@Override
	public int getOutputSize() {
		return weights.length;
	}

	@Override
	public float[] calcOutput(float[] inp) {
		MnLinalg.mulMatVec(weights, inp, out);
		MnLinalg.add(out, bias);

		return out;
	}

	@Override
	public float[] getOutput() {
		return out;
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

	@Override
	public DenseGradient createGradient() {
		return new DenseGradient(getInputSize(), getOutputSize());
	}

	/**
	 * Randomly initializes the weights of this dense layer and assigns 0 to the
	 * biases.
	 */
	@Override
	public void initParams(Random rnd) {
		float xavierFactor = 2.0f / (getInputSize() + getOutputSize());

		MnFuncs.assignGaussian(weights, rnd);
		MnLinalg.mul(weights, xavierFactor);

		MnLinalg.assign(bias, 0);
	}

	@Override
	public void learn(Gradient grad, float negLearningRate) {
		DenseGradient grad_ = (DenseGradient) grad;

		MnLinalg.mulAdd(grad_.bias, negLearningRate, bias);
		MnLinalg.mulAdd(grad_.weights, negLearningRate, weights);
	}

	@Override
	public void calcGradient(float[] inp, float[] grad_out, Gradient grad) {
		DenseGradient grad_ = (DenseGradient) grad;

		MnLinalg.outerProduct(grad_out, inp, grad_.weights);
		MnLinalg.mulVecMat(grad_out, weights, grad_.inp);
		MnLinalg.assign(grad_out, grad_.bias);
	}

	@Override
	public String toString() {
		return toStringWithLayout();
	}

	@Override
	public String getTypeShortname() {
		return "dense";
	}

	@Override
	public String toStringWithLayout() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayout(buf);
		return buf.toString();
	}

	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		buf //
				.append(getTypeShortname()).append("[") //
				.append(weights.length).append("x").append(weights[0].length) //
				.append("]") //
				.append("*[").append(weights[0].length).append("]") //
				.append("+[").append(weights.length).append("]") //
				.append("->[").append(weights.length).append("]") //
		;

		return buf;
	}

	@Override
	public String toStringWithLayoutAndValues() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayoutAndValues(buf, 4);
		return buf.toString();
	}

	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {
		NumberStats numberStats = new NumberStats();
		numberStats.aggregate(out);
		numberStats.aggregate(weights);
		numberStats.aggregate(bias);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		String indentSpaces = StringUtil.getSpaces(indent);
		buf.append(indentSpaces);
		toStringBuilderWithLayout(buf).append("\n");

		buf.append(indentSpaces).append("    weights: ");
		MnFormat.toStringBuilder(weights, buf, indent + 13, format).append("\n");

		buf.append(indentSpaces).append("    bias   : ");
		MnFormat.toStringBuilder(bias, buf, format).append("\n");

		buf.append(indentSpaces).append("    out    : ");
		MnFormat.toStringBuilder(out, buf, format);

		return buf;
	}
}
