package de.a0h.minideeplearn.operation.gradient;

import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.mininum.MnLinalg;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

public class DenseGradient extends VectorGradient {

	public float[][] weights;

	public float[] bias;

	public DenseGradient(int inpSize, int outSize) {
		super(inpSize);

		weights = new float[outSize][inpSize];
		bias = new float[outSize];
	}

	@Override
	public void clear() {
		super.clear();

		MnLinalg.assign(weights, 0.0f);
		MnLinalg.assign(bias, 0.0f);
	}

	@Override
	public void mul(float factor) {
		MnLinalg.mul(inp, factor);
		MnLinalg.mul(weights, factor);
		MnLinalg.mul(bias, factor);
	}

	@Override
	public void add(Gradient other) {
		ensureCompatibleForAdd(other);

		DenseGradient otherDense = (DenseGradient) other;

		MnLinalg.add(inp, otherDense.inp);
		MnLinalg.add(weights, otherDense.weights);
		MnLinalg.add(bias, otherDense.bias);
	}

	@Override
	public String getTypeShortname() {
		return "dense_grad";
	}

	@Override
	public String toStringWithLayout() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayout(buf);
		return buf.toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		buf //
				.append(getTypeShortname()).append("[") //
				.append(weights.length).append("x").append(weights[0].length) //
				.append("]") //
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
		numberStats.aggregate(inp);
		numberStats.aggregate(weights);
		numberStats.aggregate(bias);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		String indentSpaces = StringUtil.getSpaces(indent);
		buf.append(indentSpaces);
		toStringBuilderWithLayout(buf).append("\n");

		buf.append(indentSpaces).append("    inp    : ");
		MnFormat.toStringBuilder(inp, buf, format).append("\n");

		buf.append(indentSpaces).append("    weights: ");
		MnFormat.toStringBuilder(weights, buf, indent + 13, format).append("\n");

		buf.append(indentSpaces).append("    bias   : ");
		MnFormat.toStringBuilder(bias, buf, format);

		return buf;
	}

	protected void ensureCompatibleForAdd(Gradient other) {
		super.ensureCompatibleForAdd(other);

		float[][] otherWeights = ((DenseGradient) other).weights;
		if (otherWeights.length != weights.length) {
			throw new IllegalArgumentException("" + //
					"other gradient's weights matrix height of " + otherWeights.length + " " + //
					"is not the same as " + //
					"this gradient's weights matrix height of " + weights.length);
		}

	}
}
