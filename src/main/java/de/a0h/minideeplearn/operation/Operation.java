package de.a0h.minideeplearn.operation;

import java.util.Random;

import de.a0h.minideeplearn.operation.gradient.Gradient;

public interface Operation {

	/**
	 * Returns the size of input vectors for this element.
	 */
	public int getInputSize();

	/**
	 * Subclasses shall return true if, and only if, they have usual vector output.
	 * That's the case for elements which are not pure loss functions, e.g. for
	 * layers and for activation functions and combined loss and activation
	 * functions.
	 */
	public boolean hasOutput();

	/**
	 * Returns the size of output vectors produced by this element.
	 */
	public int getOutputSize();

	public float[] calcOutput(float[] inp);

	/**
	 * Returns the internal array which holds the output. You should call
	 * {@link #calcOutput(float[])} in order to populate this output array with
	 * meaningful data.
	 */
	public float[] getOutput();

	public boolean hasLoss();

	public float calcLoss(float[] inp, float[] target);

	public float getLoss();

	public void learn(Gradient grad, float negLearningRate);

	/**
	 * Dual-contract method for calculating the gradients down to the gradient in
	 * respect to the input to this element.
	 * 
	 * @param upstream_grad_of_out_or_target for most element types, this is the
	 *                                       gradient in respect to the output of
	 *                                       this element. This output gradient is
	 *                                       backpropagated into this element from
	 *                                       upstream elements. for loss functions
	 *                                       and combined loss functions, it has a
	 *                                       different content: the target values
	 *                                       (aka. reality values aka. ground truth)
	 *                                       of the function to approximate.
	 */
	public void calcGradient(float[] inp, float[] target_or_upstream_grad_of_out, Gradient grad_to_be_calculated);

	public void initParams(Random rnd);

	/**
	 * Creates and returns a suitable gradient object which can hold all the
	 * gradients of this element.
	 */
	public Gradient createGradient();

	/**
	 * Returns the short name of the type of this element, e.g. "ce" for
	 * cross-entropy, "dense" for a dense layer, or "softm-ce" for a combination of
	 * softmax and cross-entropy.
	 */
	public String getTypeShortname();

	/**
	 * Returns a String containing the type name and basic format info.
	 */
	public String toStringWithLayout();

	/**
	 * Returns a String containing the type name and basic format info as well as
	 * all the information required to debug a forward calculation step.
	 */
	public String toStringWithLayoutAndValues();

	public StringBuilder toStringBuilderWithLayout(StringBuilder buf);

	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent);
}
