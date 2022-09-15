package de.a0h.minideeplearn.datasetgenerator.vectortransform;

import java.util.Random;

/**
 * A vector transformation transforms one vector to another, suitable for
 * generically creating pairs of input and output data for machine learning
 * applications. The vector sizes may or may not be fixed, depending on the type
 * of transformation. The output vector for a given input vector may or may not
 * vary randomly, depending on the type of transformation.
 */
public interface VectorTransformation {

	public void transform(float[] inputVector, float[] outputVector, Random rnd);

}
