package de.a0h.minideeplearn.operation.compiler;

import java.text.DecimalFormat;
import java.util.Random;

import de.a0h.javatemplater.JavaTemplater;
import de.a0h.javatemplater.TemplateMethod;

/**
 * Method templates which can be incorporated into generated source code.
 */
class MethodTemplates {

	/**
	 * Multiplies vector vecA with scalar sca and then adds it to vector vecB,
	 * storing the result in vecB. The vector vecA is not modified. I.e. vecB[i] +=
	 * sca * vecA[i].
	 */
	@TemplateMethod
	public static void mulAddVecSca(float[] vecA, float sca, float[] vecB, int length) {
		for (int j = 0; j < length; j++) {
			vecB[j] += sca * vecA[j];
		}
	}

	/**
	 * Multiplies matrix matA with scalar sca and then adds it to matrix matB,
	 * storing the result in matB. The matrix matA is not modified. I.e. matB[i][j]
	 * += sca * matA[i][j].
	 */
	@TemplateMethod
	public static void mulAddMatSca(float[][] matA, float sca, float[][] matB, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] rowA = matA[i];
			float[] rowB = matB[i];

			for (int j = 0; j < colCount; j++) {
				rowB[j] += sca * rowA[j];
			}
		}
	}

	/**
	 * Multiplies a vector with a scalar and overwrites the vector with the results.
	 */
	@TemplateMethod
	private static void mulVecSca(float[] vec, float sca, int length) {
		for (int i = 0; i < length; i++) {
			vec[i] *= sca;
		}
	}

	/**
	 * Multiply matrix a with scalar s, overwrite a with the results.
	 */
	@TemplateMethod
	private static void mulMatSca(float[][] mat, float sca, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] row = mat[i];

			for (int j = 0; j < colCount; j++) {
				row[j] *= sca;
			}
		}
	}

	/**
	 * Assign the specified value to every element of vector vec.
	 */
	@TemplateMethod
	private static void assignVecSca(float[] vec, float value, int length) {
		for (int i = 0; i < length; i++) {
			vec[i] = value;
		}
	}

	/**
	 * Assign the specified value to every element of matrix mat.
	 */
	@TemplateMethod
	private static void assignMatSca(float[][] mat, float value, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] row = mat[i];
			for (int j = 0; j < colCount; j++) {
				row[j] = value;
			}
		}
	}

	/**
	 * Initializes the given vector using the specified random number generator to
	 * generate a gaussian distribution.
	 */
	@TemplateMethod
	private static void assignGaussianVec(float[] vec, float stddev, Random rnd, int length) {
		for (int i = 0; i < length; i++) {
			vec[i] = stddev * (float) rnd.nextGaussian();
		}
	}

	/**
	 * matA[i][j] += matB[i][j]
	 */
	@TemplateMethod
	private static void addMat(float[][] matA, float[][] matB, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] rowA = matA[i];
			float[] rowB = matB[i];

			for (int j = 0; j < colCount; j++) {
				rowA[j] += rowB[j];
			}
		}
	}

	/**
	 * vecA[i] += vecB[i]
	 */
	@TemplateMethod
	private static void addVec(float[] vecA, float[] vecB, int length) {
		for (int i = 0; i < length; i++) {
			vecA[i] += vecB[i];
		}
	}

	/**
	 * Assigns values of vecA to vecB, i.e. vecB[i] = vecA[i].
	 */
	@TemplateMethod
	private static void assignVecVec(float[] vecA, float[] vecB, int length) {
		for (int i = 0; i < length; i++) {
			vecB[i] = vecA[i];
		}
	}

	/**
	 * Multiplies a row vector by a matrix and stores the result in an output
	 * vector, i.e. out = inp * mat.
	 */
	@TemplateMethod
	private static void mulVecMat(float[] inp, float[][] mat, float[] out, int rowCount, int colCount) {
		{
			for (int j = 0; j < colCount; j++) {
				out[j] = 0.0f;
			}

			for (int i = 0; i < rowCount; i++) {
				float[] row = mat[i];
				float sca = inp[i];

				for (int j = 0; j < colCount; j++) {
					out[j] += sca * row[j];
				}
			}
		}
	}

	/**
	 * Outer product of two vectors vec_u and vec_v, creating a matrix mat, i.e. mat
	 * = vec_u ⊗ vec_v, mat[i][j] = vec_u[i] * vec_v[j].
	 */
	@TemplateMethod
	private static void outerProduct(float[] vec_u, float[] vec_v, float[][] mat, int uLength, int vLength) {
		for (int i = 0; i < uLength; i++) {
			float sca = vec_u[i];
			float[] row = mat[i];

			for (int j = 0; j < vLength; j++) {
				row[j] = sca * vec_v[j];
			}
		}
	}

	/**
	 * Initializes the given matrix using the specified random number generator to
	 * generate a gaussian normal distribution.
	 * 
	 * @param stddev standard deviation of the distribution to use
	 */
	@TemplateMethod
	private static void assignGaussianMat(float[][] mat, float stddev, Random rnd, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] row = mat[i];

			for (int j = 0; j < colCount; j++) {
				row[j] = stddev * (float) rnd.nextGaussian();
			}
		}
	}

	@TemplateMethod
	private void crossEntropyLoss_twofoldClassification_mustInline(float[] predicted, float[] target, float loss) {
		// cross entropy loss for twofold classification (aka binary classification)
		if (target[0] == 1.0f) {
			loss = -(float) Math.log(predicted[0] + Float.MIN_NORMAL);
		} else if (target[0] == 0.0f) {
			loss = -(float) Math.log(1.0f - predicted[0] + Float.MIN_NORMAL);
		} else {
			loss = -target[0] * (float) Math.log(predicted[0] + Float.MIN_NORMAL) //
					- (1.0f - target[0]) * (float) Math.log(1.0f - predicted[0] + Float.MIN_NORMAL);
		}
	}

	@TemplateMethod
	private void crossEntropyLoss_manifoldClassification_mustInline(float[] predicted, float[] target, int length,
			float loss) {
		// cross entropy loss for manifold classification (aka multinomial
		// classification)
		loss = 0.0f;

		for (int i = 0; i < length; i++) {
			if (target[i] == 0.0f) {
				continue;
			}

			loss -= target[i] * (float) Math.log(predicted[i] + Float.MIN_NORMAL);
		}

		loss /= length;
	}

	/**
	 * Calculates the gradient of a cross entropy loss in respect to the predicted
	 * probabilities (= output values of a classification net), for a twofold
	 * classification (aka binary classification).
	 * 
	 * @param out      predicted probability of category 0
	 * @param target   target probability of category 0 (usually 0.0 or 1.0)
	 * @param grad_out result which is calculated: the gradient of the loss in
	 *                 respect to the predicted probability of category 0
	 */
	@TemplateMethod
	private void crossEntropyLossGradient_twofoldClassification(float[] out, float[] target, float[] grad_out) {
		// gradient for cross entropy loss for twofold classification (aka binary
		// classification)
		grad_out[0] = (1.0f - target[0]) / (1.0f - out[0]) - target[0] / out[0];
	}

	/**
	 * Calculates the gradient of a cross entropy loss in respect to the predicted
	 * probabilities (= output values of a classification net), for a manifold
	 * classification (aka multinomial classification).
	 * 
	 * @param out      predicted probabilities, one for each category
	 * @param target   target probabilities, one for each category, usually a
	 *                 onehot-vector
	 * @param length   length of all the vectors
	 * @param grad_out result which is calculated: the gradient of the loss in
	 *                 respect to the predicted probabilities
	 */
	@TemplateMethod
	private void crossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,
			float[] grad_out) {
		// gradient for cross entropy loss for manifold classification (aka multinomial
		// classification)
		float negNReciprocal = -1.0f / length;

		for (int i = 0; i < length; i++) {
			grad_out[i] = negNReciprocal * (target[i] / out[i]);
		}
	}

	/**
	 * Calculates the gradient of a combined sigmoid with cross entropy loss in
	 * respect to the values before the application of the sigmoid (= raw output
	 * values of a classification net before applying the sigmoid activation
	 * function), for a twofold classification (aka binary classification).
	 * 
	 * @param out      predicted probability of category 0
	 * @param target   target probability of category 0 (usually 0.0 or 1.0)
	 * @param grad_inp result which is calculated: the gradient of the loss in
	 *                 respect to the values before the application of the sigmoid
	 *                 of category 0
	 */
	@TemplateMethod
	private void sigmoidWithCrossEntropyLossGradient_twofoldClassification(float[] out, float[] target,
			float[] grad_inp) {
		// grad_out is skipped; grad_inp is calculated directly:
		// gradient for sigmoid with cross entropy loss for twofold classification (aka
		// binary classification)
		grad_inp[0] = out[0] - target[0];
	}

	/**
	 * Calculates the gradient of a combined sigmoid with cross entropy loss in
	 * respect to the values before the application of the sigmoid (= raw output
	 * values of a classification net before applying the sigmoid activation
	 * function), for a manifold classification (aka multinomial classification).
	 * 
	 * <p>
	 * This classification is usually not used. Instead, in such a case softmax with
	 * cross entropy is used, because it is the natural extension of sigmoid in the
	 * case of manifold classification (aka multinomial classification).
	 * </p>
	 * 
	 * @param out      predicted probabilities, one for each category
	 * @param target   target probabilities, one for each category, usually a
	 *                 onehot-vector
	 * @param length   length of all the vectors
	 * @param grad_inp result which is calculated: the gradient of the loss in
	 *                 respect to the values before the application of the sigmoid
	 */
	@TemplateMethod
	private void sigmoidWithCrossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,
			float[] grad_inp) {
		// grad_out is skipped; grad_inp is calculated directly:
		// gradient for sigmoid with cross entropy loss for manifold classification (aka
		// multinomial classification)
		float nReciprocal = 1.0f / length;

		for (int i = 0; i < length; i++) {
			grad_inp[i] = nReciprocal * target[i] * (out[i] - 1.0f);
		}
	}

	/**
	 * Calculates the gradient of a combined softmax with cross entropy loss in
	 * respect to the values before the softmax application (= raw output values of
	 * a classification net before applying the softmax activation function), for a
	 * manifold classification (aka multinomial classification).
	 * 
	 * <p>
	 * If you have just one output probability value, softmax doesn't make sense. In
	 * this case, sigmoid serves the same purpose for twofold classification (aka
	 * binary classification) as softmax does for manifold classification (aka
	 * multinomial classification).
	 * </p>
	 * 
	 * @param out      predicted probabilities, one for each category
	 * @param target   target probabilities, one for each category, usually a
	 *                 onehot-vector
	 * @param length   length of all the vectors
	 * @param grad_inp result which is calculated: the gradient of the loss in
	 *                 respect to the values before the application of softmax
	 */
	@TemplateMethod
	private void softmaxWithCrossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,
			float[] grad_inp) {
		// grad_out is skipped; grad_inp is calculated directly:
		// gradient for softmax with cross entropy loss for manifold classification (aka
		// multinomial classification)
		float nReciprocal = 1.0f / length;

		for (int i = 0; i < length; i++) {
			grad_inp[i] = nReciprocal * (out[i] - target[i]);
		}
	}

	/**
	 * Multiplies matrix mat with vector inp, adds the bias vector and stores the
	 * result in out, i.e. calculates out = mat * inp + bias.
	 */
	@TemplateMethod
	private static void mulMatVecPlusBias(float[][] mat, float[] inp, float[] bias, float[] out, int lengthOut,
			int lengthInp) {
		for (int i = 0; i < lengthOut; i++) {
			float[] row = mat[i];

			float tmp = 0.0f;

			for (int j = 0; j < lengthInp; j++) {
				tmp += row[j] * inp[j];
			}

			out[i] = tmp + bias[i];
		}
	}

	@TemplateMethod
	private static void tanhVec(float[] inp, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			out[i] = (float) Math.tanh(inp[i]);
		}
	}

	@TemplateMethod
	private static void tanhDerivativeVec(float[] grad_inp, float[] out, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = (1.0f - out[i] * out[i]) * grad_out[i];
		}
	}

	@TemplateMethod
	private static void sigmoidVec(float[] inp, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			out[i] = 1.0f / (1.0f + (float) Math.exp(-inp[i]));
		}
	}

	@TemplateMethod
	private static void sigmoidDerivativeVec(float[] grad_inp, float[] out, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = (out[i] * (1.0f - out[i])) * grad_out[i];
		}
	}

	@TemplateMethod
	private static void softmaxVec(float[] inp, float[] out, int length) {
		{
			float max = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < length; i++) {
				max = Math.max(max, inp[i]);
			}

			float sum = 0;
			for (int i = 0; i < length; i++) {
				out[i] = (float) Math.exp(inp[i] - max);
				sum += out[i];
			}

			for (int i = 0; i < length; i++) {
				out[i] /= sum;
			}
		}
	}

	@TemplateMethod
	private static void softmaxDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {
		if (Math.random() <= 1.0f) {
			throw new UnsupportedOperationException(
					"sole softmax derivative is not supported. Use a combined softmax instead, e.g. SoftmaxWithCrossEntropyLoss.");
		}
	}

	@TemplateMethod
	private static void softplusVec(float[] inp, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			out[i] = (float) Math.log1p(Math.exp(inp[i]));
		}
	}

	@TemplateMethod
	private static void softplusDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = grad_out[i] / (1.0f + (float) Math.exp(-inp[i]));
		}
	}

	@TemplateMethod
	private static void reluVec(float[] inp, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			out[i] = Math.max(0, inp[i]);
		}
	}

	@TemplateMethod
	private static void reluDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = (inp[i] >= 0) ? grad_out[i] : 0.0f;
		}
	}

	@TemplateMethod
	private static void swishVec(float[] inp, float[] sig, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			sig[i] = 1.0f / (1.0f + (float) Math.exp(-inp[i]));
			out[i] = sig[i] * inp[i];
		}
	}

	@TemplateMethod
	private static void swishDerivativeVec(float[] grad_inp, float[] out, float[] sig, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = (out[i] + sig[i] * (1.0f - out[i])) * grad_out[i];
		}
	}

	protected static final String SPACES = "                                ";

	@TemplateMethod
	protected static StringBuilder toStringBuilderMat(float[][] mat, StringBuilder buf, int indentSpaces,
			DecimalFormat format) {
		buf.append("[");

		String indent = SPACES.substring(0, indentSpaces + 1);

		for (int i = 0; i < mat.length; i++) {
			if (i > 0) {
				buf.append(indent);
			}

			toStringBuilderVec(mat[i], buf, format);

			if (i < mat.length - 1) {
				buf.append("\n");
			}
		}

		buf.append("]");

		return buf;
	}

	@TemplateMethod
	protected static StringBuilder toStringBuilderVec(float[] vec, StringBuilder buf, DecimalFormat format) {
		buf.append("[");

		for (int i = 0; i < vec.length; i++) {
			toStringBuilderSca(vec[i], buf, format);

			if (i < vec.length - 1) {
				buf.append(" ");
			}
		}

		buf.append("]");

		return buf;
	}

	@TemplateMethod
	protected static StringBuilder toStringBuilderSca(float sca, StringBuilder buf, DecimalFormat decimalFormat) {
		buf.append(decimalFormat.format(sca));

		return buf;
	}

	@TemplateMethod
	public static void throwLossNotProvidedException(String className) {
		throw new UnsupportedOperationException("" + //
				"loss not provided: don't call this method on operations " + //
				"of type " + className + " or their compiled versions, " + //
				"because they don't provide a loss");
	}

	@TemplateMethod
	public static void throwOutputNotProvidedException(String className) {
		throw new UnsupportedOperationException("" + //
				"output not provided: don't call this method on operations " + //
				"of type " + className + " or their compiled versions, " + //
				"because they don't provide an output");
	}

	public static final void main(String[] args) throws Exception {
		JavaTemplater.generateAccessibleSource(MethodTemplates.class.getName());
	}
}
