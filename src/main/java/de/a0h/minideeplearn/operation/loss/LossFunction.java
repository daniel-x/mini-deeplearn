package de.a0h.minideeplearn.operation.loss;

import de.a0h.minideeplearn.operation.Operation;

public interface LossFunction extends Operation {

	/**
	 * Implementations of loss functions shall return true from this method.
	 */
	@Override
	public abstract boolean hasLoss();

	/**
	 * Calculates and returns the loss.
	 */
	@Override
	public abstract float calcLoss(float[] inp, float[] target);

	/**
	 * Returns the loss previously calculated in
	 * {@link #calcForwardLoss(float[], float[])}.
	 */
	@Override
	public abstract float getLoss();

}
