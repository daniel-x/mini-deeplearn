package de.a0h.minideeplearn.datasetgenerator.vectortransform;

import de.a0h.mininum.MnFuncs;

/**
 * Outputs [1, 0] when the given input vector is inside the centered sphere with
 * the given radius, and [0, 1] otherwise.
 */
public class InsideCenteredSphere extends BinaryClassificationVectorTransformation {

	float radiusSq;

	public InsideCenteredSphere(float radius) {
		this.radiusSq = radius * radius;
	}

	@Override
	protected boolean isInCategory0(float[] x) {
		return MnFuncs.norm2Squared(x) < radiusSq;
	}

}