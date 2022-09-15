package de.a0h.minideeplearn.operation.composite;

import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.NoSuchElementException;

import de.a0h.minideeplearn.operation.Operation;

public class ChainIterator implements Iterator<Operation> {

	ArrayDeque<Iterator<Operation>> stack = new ArrayDeque<>();

	public ChainIterator(Chain chain) {
		stack.add(chain.list.iterator());
	}

	@Override
	public boolean hasNext() {
		while (stack.size() > 0) {
			Iterator<Operation> subIter = stack.getLast();

			if (subIter.hasNext()) {
				return true;
			}

			stack.removeLast();
		}

		return false;
	}

	@Override
	public Operation next() {
		Iterator<Operation> subIter;

		outerLoop: while (stack.size() > 0) {
			innerLoop: while (true) {
				subIter = stack.getLast();

				if (!subIter.hasNext()) {
					stack.removeLast();
					continue outerLoop;
				}

				Operation subOp = subIter.next();

				if (subOp instanceof Chain) {
					Chain subChain = (Chain) subOp;
					Iterator<Operation> subSubIter = subChain.list.iterator();
					stack.push(subSubIter);
					continue innerLoop;
				}

				return subOp;
			}
		}

		throw new NoSuchElementException("no more elements in this iteration");
	}
}
