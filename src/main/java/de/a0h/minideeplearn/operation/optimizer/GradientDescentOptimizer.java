package de.a0h.minideeplearn.operation.optimizer;

import java.util.Random;

import de.a0h.minideeplearn.Stats;
import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.mininum.MnArrays;

public class GradientDescentOptimizer implements Optimizer {

	@Override
	public Stats run( //
			Operation model, //
			float[][] inp, //
			float[][] target, //
			int batchSize, //
			float learningRate, //
			Random shufflingRnd //
	) {
		if (inp.length != target.length) {
			throw new IllegalArgumentException("" + //
					"inp and target must be of equal length, but they are " + //
					"inp.length = " + inp.length + " and " + //
					"target.length = " + target.length);
		}

		if (batchSize == -1) {
			batchSize = inp.length;
		}

		Stats stats = new Stats(model.getOutputSize());

		Gradient sampleGrad = model.createGradient();
		Gradient batchGrad = model.createGradient();

		int[] shuffledIndices;
		if (shufflingRnd == null || batchSize == inp.length) {
			shuffledIndices = null;
		} else {
			shuffledIndices = MnArrays.generateShuffle(inp.length, shufflingRnd);
		}

		for (int i = 0; i < inp.length;) {
			int currBatchSize = Math.min(batchSize, inp.length - i);
			int currBatchEnd = i + currBatchSize;

			batchGrad.clear();
			for (; i < currBatchEnd; i++) {
				int idx = (shuffledIndices == null) ? i : shuffledIndices[i];

				float[] x = inp[idx];
				float[] t = target[idx];

				model.calcOutput(x);
				float loss = model.calcLoss(x, t);

				model.calcGradient(x, t, sampleGrad);

//				MdlOperationConfig.stringConversionPrecision = StringConversionPrecision.EXACT;
//				System.out.println(sampleGrad);

				batchGrad.add(sampleGrad);

				stats.aggregate(0, 0, loss);
			}

			model.learn(batchGrad, -learningRate / currBatchSize);
		}

		return stats;
	}
}