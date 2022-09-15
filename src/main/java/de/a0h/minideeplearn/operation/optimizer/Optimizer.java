package de.a0h.minideeplearn.operation.optimizer;

import java.util.Random;

import de.a0h.minideeplearn.Stats;
import de.a0h.minideeplearn.operation.Operation;

public interface Optimizer {

	/**
	 * Trains on all provided samples. This is usually used to train one epoch. If
	 * shufflingRnd is null or if batchSize = inp.length, then no random reordering
	 * is used. <code>batchSize = -1</code> can be used as shorthand for
	 * <code>batchSize = inp.length</code>.
	 */
	public Stats run( //
			Operation model, //
			float[][] inp, //
			float[][] target, //
			int batchSize, //
			float learningRate, //
			Random shufflingRnd //
	);
}