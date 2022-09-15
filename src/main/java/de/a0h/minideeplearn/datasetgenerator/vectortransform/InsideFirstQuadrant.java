package de.a0h.minideeplearn.datasetgenerator.vectortransform;

/**
 * Outputs [1, 0] when the given input vector is inside the first quadrant, and
 * [0, 1] otherwise.
 */
public class InsideFirstQuadrant extends BinaryClassificationVectorTransformation {

	protected boolean isInCategory0(float[] x) {
		boolean result = true;

		for (int i = 0; i < x.length; i++) {
			result &= x[i] > 0;
		}

		return result;
	}
}