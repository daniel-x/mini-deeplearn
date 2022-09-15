package de.a0h.minideeplearn.datasetgenerator.vectordistribution;

import java.util.Random;

/**
 * Generates vectors suitable for input vectors for machine learning example
 * data sets.
 */
public interface VectorDistribution {

	/**
	 * Implementations shall use rnd to fill the vector x with a random sample drawn
	 * from this distribution.
	 */
	public abstract void drawSample(float[] x, Random rnd);

}
