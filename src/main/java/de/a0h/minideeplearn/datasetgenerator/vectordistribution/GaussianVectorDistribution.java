package de.a0h.minideeplearn.datasetgenerator.vectordistribution;

import java.util.Random;

import de.a0h.mininum.MnFuncs;

public class GaussianVectorDistribution implements VectorDistribution {

	@Override
	public void drawSample(float[] x, Random rnd) {
		MnFuncs.assignGaussian(x, rnd);
	}
}