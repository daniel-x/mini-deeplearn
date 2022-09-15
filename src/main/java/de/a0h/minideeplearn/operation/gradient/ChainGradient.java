package de.a0h.minideeplearn.operation.gradient;

import java.util.ArrayList;

import de.a0h.mininum.format.StringUtil;

public class ChainGradient implements Gradient {

	ArrayList<Gradient> list = new ArrayList<>();

	public ChainGradient() {
	}

	@Override
	public String getTypeShortname() {
		return "chain_grad";
	}

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
		buf.append(getTypeShortname()).append("(");

		for (int i = 0; i < list.size(); i++) {
			Gradient el = list.get(i);

			buf.append(el.toStringWithLayout());

			buf.append(", ");
		}

		if (list.size() > 0) {
			buf.setLength(buf.length() - 2);
		}

		buf.append(")");

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

		toStringBuilderWithLayout(buf);

		if (list.size() > 0) {
			buf.append(":\n");

			indent += 4;
			for (int i = 0; i < list.size(); i++) {
				Gradient el = list.get(i);
				el.toStringBuilderWithLayoutAndValues(buf, indent).append("\n");
			}

			buf.setLength(buf.length() - 1);
		}

		return buf;
	}

	@Override
	public float[] getInputGrad() {
		ensureNotEmpty();

		return list.get(0).getInputGrad();
	}

	protected void ensureNotEmpty() {
		if (list.size() == 0) {
			throw new IllegalStateException("" + //
					"this type of gradient doesn't have any elements. " + //
					"you must first add elements to this type of gradient " + //
					"before you can call this method");
		}
	}

	@Override
	public void clear() {
		for (int i = 0; i < list.size(); i++) {
			Gradient grad = list.get(i);
			grad.clear();
		}
	}

	@Override
	public void mul(float factor) {
		for (int i = 0; i < list.size(); i++) {
			Gradient grad = list.get(i);
			grad.mul(factor);
		}
	}

	@Override
	public void add(Gradient other) {
		shallowEnsureCompatibleForAdd(other);

		ChainGradient other_ = (ChainGradient) other;

		for (int i = 0; i < list.size(); i++) {
			Gradient el = this.list.get(i);
			Gradient otherEl = other_.list.get(i);

			el.add(otherEl);
		}
	}

	public void addElement(Gradient subGrad) {
		list.add(subGrad);
	}

	public Gradient getElement(int index) {
		return list.get(index);
	}

	public int size() {
		return list.size();
	}

	protected void shallowEnsureCompatibleForAdd(Gradient other) {
		if (!getClass().equals(other.getClass())) {
			throw new IllegalArgumentException("" + //
					"other gradient of type " + other.getClass().getSimpleName() + " " + //
					"is not the same gradient type as " + //
					"this gradient, which is a " + getClass().getSimpleName());
		}

		ArrayList<Gradient> otherList = ((ChainGradient) other).list;
		if (otherList.size() != list.size()) {
			throw new IllegalArgumentException("" + //
					"other gradient has " + otherList.size() + " sub gradients, " + //
					"which is incompatible to this gradient, which " + //
					"has " + list.size() + " sub gradients");
		}
	}
}