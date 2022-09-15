package de.a0h.minideeplearn.datasetgenerator.vectordistribution;

import java.util.Random;

import de.a0h.mininum.MnFuncs;

public class UniformVectorDistribution implements VectorDistribution {

	float mMin;
	float mMax;

	public UniformVectorDistribution(float min, float max) {
		mMin = min;
		mMax = max;
	}

	@Override
	public void drawSample(float[] x, Random rnd) {
		MnFuncs.assignUniform(x, mMin, mMax, rnd);
	}
}