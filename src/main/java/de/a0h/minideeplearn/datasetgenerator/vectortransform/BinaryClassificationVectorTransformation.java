package de.a0h.minideeplearn.datasetgenerator.vectortransform;

import java.util.Arrays;
import java.util.Random;

public abstract class BinaryClassificationVectorTransformation implements VectorTransformation {

	@Override
	public void transform(float[] x, float[] y, Random ignored) {
		boolean inCategory0 = isInCategory0(x);

		y[0] = inCategory0 ? 1 : 0;
		if (y.length >= 2) {
			y[1] = 1 - y[0];

			if (y.length >= 3) {
				Arrays.fill(y, 2, y.length - 2, 0.0f);
			}
		}
	}

	/**
	 * Returns true if, and only if, the specified vector is mapped to category 0.
	 * If the return value is false, it means that the vector is mapped to category
	 * 1. The result may have random noise.
	 */
	protected abstract boolean isInCategory0(float[] x);
}
