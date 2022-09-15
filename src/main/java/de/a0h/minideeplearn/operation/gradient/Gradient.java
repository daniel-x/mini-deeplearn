package de.a0h.minideeplearn.operation.gradient;

public interface Gradient {

	/**
	 * This gradient belongs to some element/operation for which this method returns
	 * the gradient of some loss function in respect to the input vector to this
	 * element/operation.
	 */
	public float[] getInputGrad();

	/**
	 * Sets all values of this gradient object to 0.
	 */
	public void clear();

	/**
	 * Adds the values of the other gradient to this gradient. If the gradients are
	 * incompatible for this operation, an exception is thrown. They are compatible
	 * if the other gradient has the same format, i.e. the same type and sizes, as
	 * this gradient.
	 */
	public void add(Gradient other);

	/**
	 * Shall throw an IllegalArgumentException if the other gradient is not
	 * compatible for the add method.
	 * 
	 * <p>
	 * Buh buh no protected methods in java interfaces...
	 * </p>
	 */
	// protected void ensureCompatibleForAdd(Gradient other);

	/**
	 * Multiplies all values in this gradient by the specified factor.
	 */
	public void mul(float factor);

	public String getTypeShortname();

	public String toStringWithLayout();

	public StringBuilder toStringBuilderWithLayout(StringBuilder buf);

	public String toStringWithLayoutAndValues();

	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent);
}
