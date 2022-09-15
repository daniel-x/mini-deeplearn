package de.a0h.minideeplearn.operation.compiler;

import java.util.Arrays;
import java.util.HashMap;

import de.a0h.javatemplater.MethodSourceTemplate;
import de.a0h.javatemplater.MethodSourceTemplate.Param;

/**
 * Method templates which can be incorporated into generated source code.
 */
class MethodTemplatesAccessible {

	/**
	 * This method returns a map from method names to method sources.<br/>
	 * The map is generated automatically based on the source of the class<br/>
	 * MethodTemplates.
	 */
	public static HashMap<String, MethodSourceTemplate> getMethods() {
		HashMap<String, MethodSourceTemplate> result = new HashMap<>();
		result.put( //
				"mulAddVecSca", //
				new MethodSourceTemplate( //
						"mulAddVecSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiplies vector vecA with scalar sca and then adds it to vector vecB,\n" + //
						"	 * storing the result in vecB. The vector vecA is not modified. I.e. vecB[i] +=\n" + //
						"	 * sca * vecA[i].\n" + //
						"	 */\n", //
						"	public static void mulAddVecSca(float[] vecA, float sca, float[] vecB, int length) {\n", //
						"		for (int j = 0; j < length; j++) {\n" + //
						"			vecB[j] += sca * vecA[j];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vecA", "float[]"), //
								new Param("sca", "float"), //
								new Param("vecB", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"mulAddMatSca", //
				new MethodSourceTemplate( //
						"mulAddMatSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiplies matrix matA with scalar sca and then adds it to matrix matB,\n" + //
						"	 * storing the result in matB. The matrix matA is not modified. I.e. matB[i][j]\n" + //
						"	 * += sca * matA[i][j].\n" + //
						"	 */\n", //
						"	public static void mulAddMatSca(float[][] matA, float sca, float[][] matB, int rowCount, int colCount) {\n", //
						"		for (int i = 0; i < rowCount; i++) {\n" + //
						"			float[] rowA = matA[i];\n" + //
						"			float[] rowB = matB[i];\n" + //
						"\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				rowB[j] += sca * rowA[j];\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("matA", "float[][]"), //
								new Param("sca", "float"), //
								new Param("matB", "float[][]"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"mulVecSca", //
				new MethodSourceTemplate( //
						"mulVecSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiplies a vector with a scalar and overwrites the vector with the results.\n" + //
						"	 */\n", //
						"	private static void mulVecSca(float[] vec, float sca, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			vec[i] *= sca;\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vec", "float[]"), //
								new Param("sca", "float"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"mulMatSca", //
				new MethodSourceTemplate( //
						"mulMatSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiply matrix a with scalar s, overwrite a with the results.\n" + //
						"	 */\n", //
						"	private static void mulMatSca(float[][] mat, float sca, int rowCount, int colCount) {\n", //
						"		for (int i = 0; i < rowCount; i++) {\n" + //
						"			float[] row = mat[i];\n" + //
						"\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				row[j] *= sca;\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("mat", "float[][]"), //
								new Param("sca", "float"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"assignVecSca", //
				new MethodSourceTemplate( //
						"assignVecSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Assign the specified value to every element of vector vec.\n" + //
						"	 */\n", //
						"	private static void assignVecSca(float[] vec, float value, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			vec[i] = value;\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vec", "float[]"), //
								new Param("value", "float"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"assignMatSca", //
				new MethodSourceTemplate( //
						"assignMatSca", //
						"\n" + //
						"	/**\n" + //
						"	 * Assign the specified value to every element of matrix mat.\n" + //
						"	 */\n", //
						"	private static void assignMatSca(float[][] mat, float value, int rowCount, int colCount) {\n", //
						"		for (int i = 0; i < rowCount; i++) {\n" + //
						"			float[] row = mat[i];\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				row[j] = value;\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("mat", "float[][]"), //
								new Param("value", "float"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"assignGaussianVec", //
				new MethodSourceTemplate( //
						"assignGaussianVec", //
						"\n" + //
						"	/**\n" + //
						"	 * Initializes the given vector using the specified random number generator to\n" + //
						"	 * generate a gaussian distribution.\n" + //
						"	 */\n", //
						"	private static void assignGaussianVec(float[] vec, float stddev, Random rnd, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			vec[i] = stddev * (float) rnd.nextGaussian();\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vec", "float[]"), //
								new Param("stddev", "float"), //
								new Param("rnd", "Random"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"addMat", //
				new MethodSourceTemplate( //
						"addMat", //
						"\n" + //
						"	/**\n" + //
						"	 * matA[i][j] += matB[i][j]\n" + //
						"	 */\n", //
						"	private static void addMat(float[][] matA, float[][] matB, int rowCount, int colCount) {\n", //
						"		for (int i = 0; i < rowCount; i++) {\n" + //
						"			float[] rowA = matA[i];\n" + //
						"			float[] rowB = matB[i];\n" + //
						"\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				rowA[j] += rowB[j];\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("matA", "float[][]"), //
								new Param("matB", "float[][]"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"addVec", //
				new MethodSourceTemplate( //
						"addVec", //
						"\n" + //
						"	/**\n" + //
						"	 * vecA[i] += vecB[i]\n" + //
						"	 */\n", //
						"	private static void addVec(float[] vecA, float[] vecB, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			vecA[i] += vecB[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vecA", "float[]"), //
								new Param("vecB", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"assignVecVec", //
				new MethodSourceTemplate( //
						"assignVecVec", //
						"\n" + //
						"	/**\n" + //
						"	 * Assigns values of vecA to vecB, i.e. vecB[i] = vecA[i].\n" + //
						"	 */\n", //
						"	private static void assignVecVec(float[] vecA, float[] vecB, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			vecB[i] = vecA[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vecA", "float[]"), //
								new Param("vecB", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"mulVecMat", //
				new MethodSourceTemplate( //
						"mulVecMat", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiplies a row vector by a matrix and stores the result in an output\n" + //
						"	 * vector, i.e. out = inp * mat.\n" + //
						"	 */\n", //
						"	private static void mulVecMat(float[] inp, float[][] mat, float[] out, int rowCount, int colCount) {\n", //
						"		{\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				out[j] = 0.0f;\n" + //
						"			}\n" + //
						"\n" + //
						"			for (int i = 0; i < rowCount; i++) {\n" + //
						"				float[] row = mat[i];\n" + //
						"				float sca = inp[i];\n" + //
						"\n" + //
						"				for (int j = 0; j < colCount; j++) {\n" + //
						"					out[j] += sca * row[j];\n" + //
						"				}\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("mat", "float[][]"), //
								new Param("out", "float[]"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"outerProduct", //
				new MethodSourceTemplate( //
						"outerProduct", //
						"\n" + //
						"	/**\n" + //
						"	 * Outer product of two vectors vec_u and vec_v, creating a matrix mat, i.e. mat\n" + //
						"	 * = vec_u ⊗ vec_v, mat[i][j] = vec_u[i] * vec_v[j].\n" + //
						"	 */\n", //
						"	private static void outerProduct(float[] vec_u, float[] vec_v, float[][] mat, int uLength, int vLength) {\n", //
						"		for (int i = 0; i < uLength; i++) {\n" + //
						"			float sca = vec_u[i];\n" + //
						"			float[] row = mat[i];\n" + //
						"\n" + //
						"			for (int j = 0; j < vLength; j++) {\n" + //
						"				row[j] = sca * vec_v[j];\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vec_u", "float[]"), //
								new Param("vec_v", "float[]"), //
								new Param("mat", "float[][]"), //
								new Param("uLength", "int"), //
								new Param("vLength", "int") //
						) //
				) //
		);
		result.put( //
				"assignGaussianMat", //
				new MethodSourceTemplate( //
						"assignGaussianMat", //
						"\n" + //
						"	/**\n" + //
						"	 * Initializes the given matrix using the specified random number generator to\n" + //
						"	 * generate a gaussian normal distribution.\n" + //
						"	 * \n" + //
						"	 * @param stddev standard deviation of the distribution to use\n" + //
						"	 */\n", //
						"	private static void assignGaussianMat(float[][] mat, float stddev, Random rnd, int rowCount, int colCount) {\n", //
						"		for (int i = 0; i < rowCount; i++) {\n" + //
						"			float[] row = mat[i];\n" + //
						"\n" + //
						"			for (int j = 0; j < colCount; j++) {\n" + //
						"				row[j] = stddev * (float) rnd.nextGaussian();\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("mat", "float[][]"), //
								new Param("stddev", "float"), //
								new Param("rnd", "Random"), //
								new Param("rowCount", "int"), //
								new Param("colCount", "int") //
						) //
				) //
		);
		result.put( //
				"crossEntropyLoss_twofoldClassification", //
				new MethodSourceTemplate( //
						"crossEntropyLoss_twofoldClassification", //
						"\n", //
						"	private void crossEntropyLoss_twofoldClassification_mustInline(float[] predicted, float[] target, float loss) {\n", //
						"		// cross entropy loss for twofold classification (aka binary classification)\n" + //
						"		if (target[0] == 1.0f) {\n" + //
						"			loss = -(float) Math.log(predicted[0] + Float.MIN_NORMAL);\n" + //
						"		} else if (target[0] == 0.0f) {\n" + //
						"			loss = -(float) Math.log(1.0f - predicted[0] + Float.MIN_NORMAL);\n" + //
						"		} else {\n" + //
						"			loss = -target[0] * (float) Math.log(predicted[0] + Float.MIN_NORMAL) //\n" + //
						"					- (1.0f - target[0]) * (float) Math.log(1.0f - predicted[0] + Float.MIN_NORMAL);\n" + //
						"		}\n", //
						true, //
						Arrays.<Param>asList( //
								new Param("predicted", "float[]"), //
								new Param("target", "float[]"), //
								new Param("loss", "float") //
						) //
				) //
		);
		result.put( //
				"crossEntropyLoss_manifoldClassification", //
				new MethodSourceTemplate( //
						"crossEntropyLoss_manifoldClassification", //
						"\n", //
						"	private void crossEntropyLoss_manifoldClassification_mustInline(float[] predicted, float[] target, int length,\n" + //
						"			float loss) {\n", //
						"		// cross entropy loss for manifold classification (aka multinomial\n" + //
						"		// classification)\n" + //
						"		loss = 0.0f;\n" + //
						"\n" + //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			if (target[i] == 0.0f) {\n" + //
						"				continue;\n" + //
						"			}\n" + //
						"\n" + //
						"			loss -= target[i] * (float) Math.log(predicted[i] + Float.MIN_NORMAL);\n" + //
						"		}\n" + //
						"\n" + //
						"		loss /= length;\n", //
						true, //
						Arrays.<Param>asList( //
								new Param("predicted", "float[]"), //
								new Param("target", "float[]"), //
								new Param("length", "int"), //
								new Param("loss", "float") //
						) //
				) //
		);
		result.put( //
				"crossEntropyLossGradient_twofoldClassification", //
				new MethodSourceTemplate( //
						"crossEntropyLossGradient_twofoldClassification", //
						"\n" + //
						"	/**\n" + //
						"	 * Calculates the gradient of a cross entropy loss in respect to the predicted\n" + //
						"	 * probabilities (= output values of a classification net), for a twofold\n" + //
						"	 * classification (aka binary classification).\n" + //
						"	 * \n" + //
						"	 * @param out      predicted probability of category 0\n" + //
						"	 * @param target   target probability of category 0 (usually 0.0 or 1.0)\n" + //
						"	 * @param grad_out result which is calculated: the gradient of the loss in\n" + //
						"	 *                 respect to the predicted probability of category 0\n" + //
						"	 */\n", //
						"	private void crossEntropyLossGradient_twofoldClassification(float[] out, float[] target, float[] grad_out) {\n", //
						"		// gradient for cross entropy loss for twofold classification (aka binary\n" + //
						"		// classification)\n" + //
						"		grad_out[0] = (1.0f - target[0]) / (1.0f - out[0]) - target[0] / out[0];\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("out", "float[]"), //
								new Param("target", "float[]"), //
								new Param("grad_out", "float[]") //
						) //
				) //
		);
		result.put( //
				"crossEntropyLossGradient_manifoldClassification", //
				new MethodSourceTemplate( //
						"crossEntropyLossGradient_manifoldClassification", //
						"\n" + //
						"	/**\n" + //
						"	 * Calculates the gradient of a cross entropy loss in respect to the predicted\n" + //
						"	 * probabilities (= output values of a classification net), for a manifold\n" + //
						"	 * classification (aka multinomial classification).\n" + //
						"	 * \n" + //
						"	 * @param out      predicted probabilities, one for each category\n" + //
						"	 * @param target   target probabilities, one for each category, usually a\n" + //
						"	 *                 onehot-vector\n" + //
						"	 * @param length   length of all the vectors\n" + //
						"	 * @param grad_out result which is calculated: the gradient of the loss in\n" + //
						"	 *                 respect to the predicted probabilities\n" + //
						"	 */\n", //
						"	private void crossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,\n" + //
						"			float[] grad_out) {\n", //
						"		// gradient for cross entropy loss for manifold classification (aka multinomial\n" + //
						"		// classification)\n" + //
						"		float negNReciprocal = -1.0f / length;\n" + //
						"\n" + //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_out[i] = negNReciprocal * (target[i] / out[i]);\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("out", "float[]"), //
								new Param("target", "float[]"), //
								new Param("length", "int"), //
								new Param("grad_out", "float[]") //
						) //
				) //
		);
		result.put( //
				"sigmoidWithCrossEntropyLossGradient_twofoldClassification", //
				new MethodSourceTemplate( //
						"sigmoidWithCrossEntropyLossGradient_twofoldClassification", //
						"\n" + //
						"	/**\n" + //
						"	 * Calculates the gradient of a combined sigmoid with cross entropy loss in\n" + //
						"	 * respect to the values before the application of the sigmoid (= raw output\n" + //
						"	 * values of a classification net before applying the sigmoid activation\n" + //
						"	 * function), for a twofold classification (aka binary classification).\n" + //
						"	 * \n" + //
						"	 * @param out      predicted probability of category 0\n" + //
						"	 * @param target   target probability of category 0 (usually 0.0 or 1.0)\n" + //
						"	 * @param grad_inp result which is calculated: the gradient of the loss in\n" + //
						"	 *                 respect to the values before the application of the sigmoid\n" + //
						"	 *                 of category 0\n" + //
						"	 */\n", //
						"	private void sigmoidWithCrossEntropyLossGradient_twofoldClassification(float[] out, float[] target,\n" + //
						"			float[] grad_inp) {\n", //
						"		// grad_out is skipped; grad_inp is calculated directly:\n" + //
						"		// gradient for sigmoid with cross entropy loss for twofold classification (aka\n" + //
						"		// binary classification)\n" + //
						"		grad_inp[0] = out[0] - target[0];\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("out", "float[]"), //
								new Param("target", "float[]"), //
								new Param("grad_inp", "float[]") //
						) //
				) //
		);
		result.put( //
				"sigmoidWithCrossEntropyLossGradient_manifoldClassification", //
				new MethodSourceTemplate( //
						"sigmoidWithCrossEntropyLossGradient_manifoldClassification", //
						"\n" + //
						"	/**\n" + //
						"	 * Calculates the gradient of a combined sigmoid with cross entropy loss in\n" + //
						"	 * respect to the values before the application of the sigmoid (= raw output\n" + //
						"	 * values of a classification net before applying the sigmoid activation\n" + //
						"	 * function), for a manifold classification (aka multinomial classification).\n" + //
						"	 * \n" + //
						"	 * <p>\n" + //
						"	 * This classification is usually not used. Instead, in such a case softmax with\n" + //
						"	 * cross entropy is used, because it is the natural extension of sigmoid in the\n" + //
						"	 * case of manifold classification (aka multinomial classification).\n" + //
						"	 * </p>\n" + //
						"	 * \n" + //
						"	 * @param out      predicted probabilities, one for each category\n" + //
						"	 * @param target   target probabilities, one for each category, usually a\n" + //
						"	 *                 onehot-vector\n" + //
						"	 * @param length   length of all the vectors\n" + //
						"	 * @param grad_inp result which is calculated: the gradient of the loss in\n" + //
						"	 *                 respect to the values before the application of the sigmoid\n" + //
						"	 */\n", //
						"	private void sigmoidWithCrossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,\n" + //
						"			float[] grad_inp) {\n", //
						"		// grad_out is skipped; grad_inp is calculated directly:\n" + //
						"		// gradient for sigmoid with cross entropy loss for manifold classification (aka\n" + //
						"		// multinomial classification)\n" + //
						"		float nReciprocal = 1.0f / length;\n" + //
						"\n" + //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = nReciprocal * target[i] * (out[i] - 1.0f);\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("out", "float[]"), //
								new Param("target", "float[]"), //
								new Param("length", "int"), //
								new Param("grad_inp", "float[]") //
						) //
				) //
		);
		result.put( //
				"softmaxWithCrossEntropyLossGradient_manifoldClassification", //
				new MethodSourceTemplate( //
						"softmaxWithCrossEntropyLossGradient_manifoldClassification", //
						"\n" + //
						"	/**\n" + //
						"	 * Calculates the gradient of a combined softmax with cross entropy loss in\n" + //
						"	 * respect to the values before the softmax application (= raw output values of\n" + //
						"	 * a classification net before applying the softmax activation function), for a\n" + //
						"	 * manifold classification (aka multinomial classification).\n" + //
						"	 * \n" + //
						"	 * <p>\n" + //
						"	 * If you have just one output probability value, softmax doesn't make sense. In\n" + //
						"	 * this case, sigmoid serves the same purpose for twofold classification (aka\n" + //
						"	 * binary classification) as softmax does for manifold classification (aka\n" + //
						"	 * multinomial classification).\n" + //
						"	 * </p>\n" + //
						"	 * \n" + //
						"	 * @param out      predicted probabilities, one for each category\n" + //
						"	 * @param target   target probabilities, one for each category, usually a\n" + //
						"	 *                 onehot-vector\n" + //
						"	 * @param length   length of all the vectors\n" + //
						"	 * @param grad_inp result which is calculated: the gradient of the loss in\n" + //
						"	 *                 respect to the values before the application of softmax\n" + //
						"	 */\n", //
						"	private void softmaxWithCrossEntropyLossGradient_manifoldClassification(float[] out, float[] target, int length,\n" + //
						"			float[] grad_inp) {\n", //
						"		// grad_out is skipped; grad_inp is calculated directly:\n" + //
						"		// gradient for softmax with cross entropy loss for manifold classification (aka\n" + //
						"		// multinomial classification)\n" + //
						"		float nReciprocal = 1.0f / length;\n" + //
						"\n" + //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = nReciprocal * (out[i] - target[i]);\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("out", "float[]"), //
								new Param("target", "float[]"), //
								new Param("length", "int"), //
								new Param("grad_inp", "float[]") //
						) //
				) //
		);
		result.put( //
				"mulMatVecPlusBias", //
				new MethodSourceTemplate( //
						"mulMatVecPlusBias", //
						"\n" + //
						"	/**\n" + //
						"	 * Multiplies matrix mat with vector inp, adds the bias vector and stores the\n" + //
						"	 * result in out, i.e. calculates out = mat * inp + bias.\n" + //
						"	 */\n", //
						"	private static void mulMatVecPlusBias(float[][] mat, float[] inp, float[] bias, float[] out, int lengthOut,\n" + //
						"			int lengthInp) {\n", //
						"		for (int i = 0; i < lengthOut; i++) {\n" + //
						"			float[] row = mat[i];\n" + //
						"\n" + //
						"			float tmp = 0.0f;\n" + //
						"\n" + //
						"			for (int j = 0; j < lengthInp; j++) {\n" + //
						"				tmp += row[j] * inp[j];\n" + //
						"			}\n" + //
						"\n" + //
						"			out[i] = tmp + bias[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("mat", "float[][]"), //
								new Param("inp", "float[]"), //
								new Param("bias", "float[]"), //
								new Param("out", "float[]"), //
								new Param("lengthOut", "int"), //
								new Param("lengthInp", "int") //
						) //
				) //
		);
		result.put( //
				"tanhVec", //
				new MethodSourceTemplate( //
						"tanhVec", //
						"\n", //
						"	private static void tanhVec(float[] inp, float[] out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			out[i] = (float) Math.tanh(inp[i]);\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"tanhDerivativeVec", //
				new MethodSourceTemplate( //
						"tanhDerivativeVec", //
						"\n", //
						"	private static void tanhDerivativeVec(float[] grad_inp, float[] out, float[] grad_out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = (1.0f - out[i] * out[i]) * grad_out[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"sigmoidVec", //
				new MethodSourceTemplate( //
						"sigmoidVec", //
						"\n", //
						"	private static void sigmoidVec(float[] inp, float[] out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			out[i] = 1.0f / (1.0f + (float) Math.exp(-inp[i]));\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"sigmoidDerivativeVec", //
				new MethodSourceTemplate( //
						"sigmoidDerivativeVec", //
						"\n", //
						"	private static void sigmoidDerivativeVec(float[] grad_inp, float[] out, float[] grad_out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = (out[i] * (1.0f - out[i])) * grad_out[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"softmaxVec", //
				new MethodSourceTemplate( //
						"softmaxVec", //
						"\n", //
						"	private static void softmaxVec(float[] inp, float[] out, int length) {\n", //
						"		{\n" + //
						"			float max = Float.NEGATIVE_INFINITY;\n" + //
						"			for (int i = 0; i < length; i++) {\n" + //
						"				max = Math.max(max, inp[i]);\n" + //
						"			}\n" + //
						"\n" + //
						"			float sum = 0;\n" + //
						"			for (int i = 0; i < length; i++) {\n" + //
						"				out[i] = (float) Math.exp(inp[i] - max);\n" + //
						"				sum += out[i];\n" + //
						"			}\n" + //
						"\n" + //
						"			for (int i = 0; i < length; i++) {\n" + //
						"				out[i] /= sum;\n" + //
						"			}\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"softmaxDerivativeVec", //
				new MethodSourceTemplate( //
						"softmaxDerivativeVec", //
						"\n", //
						"	private static void softmaxDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {\n", //
						"		if (Math.random() <= 1.0f) {\n" + //
						"			throw new UnsupportedOperationException(\n" + //
						"					\"sole softmax derivative is not supported. Use a combined softmax instead, e.g. SoftmaxWithCrossEntropyLoss.\");\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("inp", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"softplusVec", //
				new MethodSourceTemplate( //
						"softplusVec", //
						"\n", //
						"	private static void softplusVec(float[] inp, float[] out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			out[i] = (float) Math.log1p(Math.exp(inp[i]));\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"softplusDerivativeVec", //
				new MethodSourceTemplate( //
						"softplusDerivativeVec", //
						"\n", //
						"	private static void softplusDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = grad_out[i] / (1.0f + (float) Math.exp(-inp[i]));\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("inp", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"reluVec", //
				new MethodSourceTemplate( //
						"reluVec", //
						"\n", //
						"	private static void reluVec(float[] inp, float[] out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			out[i] = Math.max(0, inp[i]);\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"reluDerivativeVec", //
				new MethodSourceTemplate( //
						"reluDerivativeVec", //
						"\n", //
						"	private static void reluDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = (inp[i] >= 0) ? grad_out[i] : 0.0f;\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("inp", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"swishVec", //
				new MethodSourceTemplate( //
						"swishVec", //
						"\n", //
						"	private static void swishVec(float[] inp, float[] sig, float[] out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			sig[i] = 1.0f / (1.0f + (float) Math.exp(-inp[i]));\n" + //
						"			out[i] = sig[i] * inp[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("inp", "float[]"), //
								new Param("sig", "float[]"), //
								new Param("out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"swishDerivativeVec", //
				new MethodSourceTemplate( //
						"swishDerivativeVec", //
						"\n", //
						"	private static void swishDerivativeVec(float[] grad_inp, float[] out, float[] sig, float[] grad_out, int length) {\n", //
						"		for (int i = 0; i < length; i++) {\n" + //
						"			grad_inp[i] = (out[i] + sig[i] * (1.0f - out[i])) * grad_out[i];\n" + //
						"		}\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("grad_inp", "float[]"), //
								new Param("out", "float[]"), //
								new Param("sig", "float[]"), //
								new Param("grad_out", "float[]"), //
								new Param("length", "int") //
						) //
				) //
		);
		result.put( //
				"toStringBuilderMat", //
				new MethodSourceTemplate( //
						"toStringBuilderMat", //
						"\n", //
						"	protected static StringBuilder toStringBuilderMat(float[][] mat, StringBuilder buf, int indentSpaces,\n" + //
						"			DecimalFormat format) {\n", //
						"		buf.append(\"[\");\n" + //
						"\n" + //
						"		String indent = SPACES.substring(0, indentSpaces + 1);\n" + //
						"\n" + //
						"		for (int i = 0; i < mat.length; i++) {\n" + //
						"			if (i > 0) {\n" + //
						"				buf.append(indent);\n" + //
						"			}\n" + //
						"\n" + //
						"			toStringBuilderVec(mat[i], buf, format);\n" + //
						"\n" + //
						"			if (i < mat.length - 1) {\n" + //
						"				buf.append(\"\\n\");\n" + //
						"			}\n" + //
						"		}\n" + //
						"\n" + //
						"		buf.append(\"]\");\n" + //
						"\n" + //
						"		return buf;\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("mat", "float[][]"), //
								new Param("buf", "StringBuilder"), //
								new Param("indentSpaces", "int"), //
								new Param("format", "DecimalFormat") //
						) //
				) //
		);
		result.put( //
				"toStringBuilderVec", //
				new MethodSourceTemplate( //
						"toStringBuilderVec", //
						"\n", //
						"	protected static StringBuilder toStringBuilderVec(float[] vec, StringBuilder buf, DecimalFormat format) {\n", //
						"		buf.append(\"[\");\n" + //
						"\n" + //
						"		for (int i = 0; i < vec.length; i++) {\n" + //
						"			toStringBuilderSca(vec[i], buf, format);\n" + //
						"\n" + //
						"			if (i < vec.length - 1) {\n" + //
						"				buf.append(\" \");\n" + //
						"			}\n" + //
						"		}\n" + //
						"\n" + //
						"		buf.append(\"]\");\n" + //
						"\n" + //
						"		return buf;\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("vec", "float[]"), //
								new Param("buf", "StringBuilder"), //
								new Param("format", "DecimalFormat") //
						) //
				) //
		);
		result.put( //
				"toStringBuilderSca", //
				new MethodSourceTemplate( //
						"toStringBuilderSca", //
						"\n", //
						"	protected static StringBuilder toStringBuilderSca(float sca, StringBuilder buf, DecimalFormat decimalFormat) {\n", //
						"		buf.append(decimalFormat.format(sca));\n" + //
						"\n" + //
						"		return buf;\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("sca", "float"), //
								new Param("buf", "StringBuilder"), //
								new Param("decimalFormat", "DecimalFormat") //
						) //
				) //
		);
		result.put( //
				"throwLossNotProvidedException", //
				new MethodSourceTemplate( //
						"throwLossNotProvidedException", //
						"\n", //
						"	public static void throwLossNotProvidedException(String className) {\n", //
						"		throw new UnsupportedOperationException(\"\" + //\n" + //
						"				\"loss not provided: don't call this method on operations \" + //\n" + //
						"				\"of type \" + className + \" or their compiled versions, \" + //\n" + //
						"				\"because they don't provide a loss\");\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("className", "String") //
						) //
				) //
		);
		result.put( //
				"throwOutputNotProvidedException", //
				new MethodSourceTemplate( //
						"throwOutputNotProvidedException", //
						"\n", //
						"	public static void throwOutputNotProvidedException(String className) {\n", //
						"		throw new UnsupportedOperationException(\"\" + //\n" + //
						"				\"output not provided: don't call this method on operations \" + //\n" + //
						"				\"of type \" + className + \" or their compiled versions, \" + //\n" + //
						"				\"because they don't provide an output\");\n", //
						false, //
						Arrays.<Param>asList( //
								new Param("className", "String") //
						) //
				) //
		);

		return result;
	}
}
