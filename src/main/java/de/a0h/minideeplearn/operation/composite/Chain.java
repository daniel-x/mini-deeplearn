package de.a0h.minideeplearn.operation.composite;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.gradient.ChainGradient;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.mininum.format.StringUtil;

/**
 * A chained sequence of operations, which can be forward-evaluated and back
 * propagated.
 */
public class Chain implements Operation, Iterable<Operation> {

	protected ArrayList<Operation> list = new ArrayList<>();

	public Chain() {
	}

	public void add(Operation op) {
		list.add(op);
	}

	/**
	 * Returns an iterator which iterates over all sub operations and their sub
	 * operations in forward evaluation order, skipping all Chains objects and only
	 * returning elementary sub operations.
	 */
	@Override
	public Iterator<Operation> iterator() {
		return new ChainIterator(this);
	}

	/**
	 * Returns true if, and only if, this chain is flat, i.e. if this chain contains
	 * only elementary operations and not composite operations.
	 */
	public boolean isFlat() {
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof Chain) {
				return false;
			}
		}

		return true;
	}

	/**
	 * If this chain is flat, this is returned, or else a flattened version of this
	 * chain is returned containing the one and same sub operation objects.
	 * Thereafter, the elementary sub operations are contained in both chains, this
	 * one and the returned one. Modifying one also modifies the other one.
	 */
	public Chain flattened() {
		if (isFlat()) {
			return this;
		}

		Chain result = new Chain();

		for (Operation op : this) {
			result.add(op);
		}

		return result;
	}

	@Override
	public int getInputSize() {
		Operation op = getFirst();
		return op.getInputSize();
	}

	@Override
	public boolean hasOutput() {
		Operation op = getLast();
		return op.hasOutput();
	}

	@Override
	public int getOutputSize() {
		Operation op = getLast();
		return op.getOutputSize();
	}

	@Override
	public float[] getOutput() {
		Operation op = getLast();
		return op.getOutput();
	}

	@Override
	public boolean hasLoss() {
		Operation op = getLast();
		return op.hasLoss();
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		Operation op = getLast();
		float loss = op.calcLoss(inp, target);

		return loss;
	}

	@Override
	public float getLoss() {
		Operation op = getLast();
		return op.getLoss();
	}

	@Override
	public float[] calcOutput(float[] inp) {
		for (int i = 0; i < list.size(); i++) {
			Operation op = list.get(i);

			if (op.hasOutput()) {
				inp = op.calcOutput(inp);
			}
		}

		return inp;
	}

	@Override
	public void calcGradient(float[] inp, float[] target_or_upstream_grad_of_out, Gradient grad) {
		shallowEnsureCompatibleGradient(grad);

		ChainGradient grad_ = (ChainGradient) grad;

		// System.out.println("Chain.calcGradient:\n" + this);

		for (int i = list.size() - 1; i >= 0; i--) {
			Operation op = list.get(i);
			float[] prev_out;
			if (i >= 1) {
				Operation prevOp = list.get(i - 1);
				prev_out = prevOp.getOutput();
			} else {
				prev_out = inp;
			}

			Gradient grad_op = grad_.getElement(i);

			op.calcGradient(prev_out, target_or_upstream_grad_of_out, grad_op);

			target_or_upstream_grad_of_out = grad_op.getInputGrad();
		}
	}

	public void shallowEnsureCompatibleGradient(Gradient grad) {
		if (!grad.getClass().equals(ChainGradient.class)) {
			throw new IllegalArgumentException("" + //
					"gradient of type " + grad.getClass().getSimpleName() + " " + //
					"is not the same gradient type as " + //
					"this class's gradient, which is " + ChainGradient.class.getSimpleName());
		}

		ChainGradient grad_ = (ChainGradient) grad;
		if (grad_.size() != list.size()) {
			throw new IllegalArgumentException("" + //
					"gradient has " + grad_.size() + " sub gradients, " + //
					"which is incompatible to this chain, because " + //
					"the latter one has " + list.size() + " elements");
		}
	}

	@Override
	public ChainGradient createGradient() {
		ChainGradient grad = new ChainGradient();

		for (int i = 0; i < list.size(); i++) {
			Operation op = list.get(i);

			Gradient subGrad = op.createGradient();

			grad.addElement(subGrad);
		}

		return grad;
	}

	public int size() {
		return list.size();
	}

	public void set(int index, Operation op) {
		list.set(index, op);
	}

	public Operation get(int index) {
		return list.get(index);
	}

	public Operation getFirst() {
		ensureNotEmpty();
		return list.get(0);
	}

	public Operation getLast() {
		ensureNotEmpty();
		return list.get(list.size() - 1);
	}

	protected void ensureNotEmpty() {
		if (list.isEmpty()) {
			throw new IllegalStateException("" + //
					"this chain doesn't have any operations. " + //
					"you must first add operations to a chain before " + //
					"you can call this method");
		}
	}

	@Override
	public String getTypeShortname() {
		return "chain";
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
			Operation op = list.get(i);

			buf.append(op.toStringWithLayout());

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
				Operation el = list.get(i);
				el.toStringBuilderWithLayoutAndValues(buf, indent).append("\n");
			}

			buf.setLength(buf.length() - 1);
		}

		return buf;
	}

	@Override
	public void initParams(Random rnd) {
		for (int i = 0; i < list.size(); i++) {
			Operation el = list.get(i);
			el.initParams(rnd);
		}
	}

	@Override
	public void learn(Gradient grad, float negLearningRate) {
		shallowEnsureCompatibleGradient(grad);

		ChainGradient grad_ = (ChainGradient) grad;

		for (int i = 0; i < list.size(); i++) {
			Operation el = list.get(i);

			Gradient grad_el = grad_.getElement(i);

			el.learn(grad_el, negLearningRate);
		}
	}
}