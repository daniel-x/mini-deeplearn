package de.a0h.minideeplearn.operation.gradient;

import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.mininum.MnLinalg;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.MnFormat;
import de.a0h.mininum.format.NumberStats;
import de.a0h.mininum.format.StringUtil;

public class VectorGradient implements Gradient {

	public float[] inp;

	public VectorGradient(int inpSize) {
		inp = new float[inpSize];
	}

	@Override
	public float[] getInputGrad() {
		return inp;
	}

	@Override
	public void clear() {
		MnLinalg.assign(inp, 0.0f);
	}

	@Override
	public void add(Gradient other) {
		ensureCompatibleForAdd(other);

		VectorGradient otherVec = (VectorGradient) other;

		MnLinalg.add(inp, otherVec.inp);
	}

	protected void ensureCompatibleForAdd(Gradient other) {
		if (!getClass().equals(other.getClass())) {
			throw new IllegalArgumentException("" + //
					"other gradient of type " + other.getClass().getSimpleName() + " " + //
					"is not the same gradient type as " + //
					"this gradient, which is a " + getClass().getSimpleName());
		}

		float[] otherInpGrad = ((VectorGradient) other).inp;
		if (otherInpGrad.length != inp.length) {
			throw new IllegalArgumentException("" + //
					"other gradient's input length of " + otherInpGrad.length + " " + //
					"is not the same as " + //
					"this gradient's input length of " + inp.length);
		}
	}

	public void mul(float factor) {
		MnLinalg.mul(inp, factor);
	}

	@Override
	public String getTypeShortname() {
		return "vector_grad";
	}

	@Override
	public String toString() {
		return toStringWithLayoutAndValues();
	}

	@Override
	public String toStringWithLayout() {
		StringBuilder buf = new StringBuilder();
		toStringBuilderWithLayout(buf);
		return buf.toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		buf.append(getTypeShortname()).append("[").append(inp.length).append("]");

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
		buf.append(StringUtil.getSpaces(indent));

		toStringBuilderWithLayout(buf).append(": inp: ");

		NumberStats numberStats = new NumberStats();
		numberStats.aggregate(inp);
		DecimalFormatWithPadding format = OperationUtil.getFormat(numberStats);

		MnFormat.toStringBuilder(inp, buf, format);

		return buf;
	}
}