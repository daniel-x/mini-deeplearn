package de.a0h.minideeplearn.operation.compiler.output;

import java.util.Arrays;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.Random;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.gradient.Gradient;

public class CompiledChain implements Operation {

	// dense[64x42]*[42]+[64]->[64]
	public final float[][] w0 = new float[64][42];
	public final float[] b0 = new float[64];
	public final float[] a01 = new float[64];

	// relu[64]
	public final float[] a02 = new float[64];

	// dense[48x64]*[64]+[48]->[48]
	public final float[][] w1 = new float[48][64];
	public final float[] b1 = new float[48];
	public final float[] a03 = new float[48];

	// relu[48]
	public final float[] a04 = new float[48];

	// dense[32x48]*[48]+[32]->[32]
	public final float[][] w2 = new float[32][48];
	public final float[] b2 = new float[32];
	public final float[] a05 = new float[32];

	// relu[32]
	public final float[] a06 = new float[32];

	// dense[24x32]*[32]+[24]->[24]
	public final float[][] w3 = new float[24][32];
	public final float[] b3 = new float[24];
	public final float[] a07 = new float[24];

	// relu[24]
	public final float[] a08 = new float[24];

	// dense[16x24]*[24]+[16]->[16]
	public final float[][] w4 = new float[16][24];
	public final float[] b4 = new float[16];
	public final float[] a09 = new float[16];

	// relu[16]
	public final float[] a10 = new float[16];

	// dense[7x16]*[16]+[7]->[7]
	public final float[][] w5 = new float[7][16];
	public final float[] b5 = new float[7];
	public final float[] a11 = new float[7];

	// softmax-ce[7]
	public final float[] a12 = new float[7];
	public float loss;

	@Override
	public void initParams(Random rnd) {
		float xavierFactor;

		// dense[64x42]*[42]+[64]->[64]
		xavierFactor = 2.0f / (42 + 64);
		assignGaussianMat(w0, xavierFactor, rnd, 64, 42);
		assignVecSca(b0, 0.0f, 64);

		// relu[64]

		// dense[48x64]*[64]+[48]->[48]
		xavierFactor = 2.0f / (64 + 48);
		assignGaussianMat(w1, xavierFactor, rnd, 48, 64);
		assignVecSca(b1, 0.0f, 48);

		// relu[48]

		// dense[32x48]*[48]+[32]->[32]
		xavierFactor = 2.0f / (48 + 32);
		assignGaussianMat(w2, xavierFactor, rnd, 32, 48);
		assignVecSca(b2, 0.0f, 32);

		// relu[32]

		// dense[24x32]*[32]+[24]->[24]
		xavierFactor = 2.0f / (32 + 24);
		assignGaussianMat(w3, xavierFactor, rnd, 24, 32);
		assignVecSca(b3, 0.0f, 24);

		// relu[24]

		// dense[16x24]*[24]+[16]->[16]
		xavierFactor = 2.0f / (24 + 16);
		assignGaussianMat(w4, xavierFactor, rnd, 16, 24);
		assignVecSca(b4, 0.0f, 16);

		// relu[16]

		// dense[7x16]*[16]+[7]->[7]
		xavierFactor = 2.0f / (16 + 7);
		assignGaussianMat(w5, xavierFactor, rnd, 7, 16);
		assignVecSca(b5, 0.0f, 7);

		// softmax-ce[7]

	}

	@Override
	public int getInputSize() {
		return 42;
	}

	@Override
	public boolean hasOutput() {
		return true;
	}

	@Override
	public int getOutputSize() {
		return 7;
	}

	@Override
	public float[] calcOutput(float[] a00) {
		// dense[64x42]*[42]+[64]->[64]
		mulMatVecPlusBias(w0, a00, b0, a01, 64, 42);

		// relu[64]
		reluVec(a01, a02, 64);

		// dense[48x64]*[64]+[48]->[48]
		mulMatVecPlusBias(w1, a02, b1, a03, 48, 64);

		// relu[48]
		reluVec(a03, a04, 48);

		// dense[32x48]*[48]+[32]->[32]
		mulMatVecPlusBias(w2, a04, b2, a05, 32, 48);

		// relu[32]
		reluVec(a05, a06, 32);

		// dense[24x32]*[32]+[24]->[24]
		mulMatVecPlusBias(w3, a06, b3, a07, 24, 32);

		// relu[24]
		reluVec(a07, a08, 24);

		// dense[16x24]*[24]+[16]->[16]
		mulMatVecPlusBias(w4, a08, b4, a09, 16, 24);

		// relu[16]
		reluVec(a09, a10, 16);

		// dense[7x16]*[16]+[7]->[7]
		mulMatVecPlusBias(w5, a10, b5, a11, 7, 16);

		// softmax-ce[7]
		softmaxVec(a11, a12, 7);

		return a12;
	}

	@Override
	public float[] getOutput() {
		return a12;
	}

	@Override
	public boolean hasLoss() {
		return true;
	}

	@Override
	public float calcLoss(float[] inp, float[] target) {
		// cross entropy loss for manifold classification (aka multinomial
		// classification)
		loss = 0.0f;

		for (int i = 0; i < 7; i++) {
			if (target[i] == 0.0f) {
				continue;
			}

			loss -= target[i] * (float) Math.log(a12[i] + Float.MIN_NORMAL);
		}

		loss /= 7;

		return loss;
	}

	@Override
	public float getLoss() {
		return loss;
	}

	@Override
	public CompiledGradient createGradient() {
		return new CompiledGradient();
	}

	@Override
	public void calcGradient( //
			float[] inp, //
			float[] target, //
			Gradient grad_ //
	) {
		CompiledGradient grad;
		try {
			grad = (CompiledGradient) grad_;
		} catch (ClassCastException e) {
			throw new IllegalArgumentException("" + //
					"grad_ must be of type " + //
					"de.a0h.minideeplearn.operation.compiler.output.CompiledChain.CompiledGradient, " + //
					"but it is a " + //
					grad_.getClass().getName(), e);
		}

		// softmax-ce[7]
		softmaxWithCrossEntropyLossGradient_manifoldClassification(a12, target, 7, grad.a11);

		// dense[7x16]*[16]+[7]->[7]
		outerProduct(grad.a11, a10, grad.w5, 7, 16);
		mulVecMat(grad.a11, w5, grad.a10, 7, 16);
		assignVecVec(grad.a11, grad.b5, 7);

		// relu[16]
		reluDerivativeVec(grad.a09, a09, grad.a10, 16);

		// dense[16x24]*[24]+[16]->[16]
		outerProduct(grad.a09, a08, grad.w4, 16, 24);
		mulVecMat(grad.a09, w4, grad.a08, 16, 24);
		assignVecVec(grad.a09, grad.b4, 16);

		// relu[24]
		reluDerivativeVec(grad.a07, a07, grad.a08, 24);

		// dense[24x32]*[32]+[24]->[24]
		outerProduct(grad.a07, a06, grad.w3, 24, 32);
		mulVecMat(grad.a07, w3, grad.a06, 24, 32);
		assignVecVec(grad.a07, grad.b3, 24);

		// relu[32]
		reluDerivativeVec(grad.a05, a05, grad.a06, 32);

		// dense[32x48]*[48]+[32]->[32]
		outerProduct(grad.a05, a04, grad.w2, 32, 48);
		mulVecMat(grad.a05, w2, grad.a04, 32, 48);
		assignVecVec(grad.a05, grad.b2, 32);

		// relu[48]
		reluDerivativeVec(grad.a03, a03, grad.a04, 48);

		// dense[48x64]*[64]+[48]->[48]
		outerProduct(grad.a03, a02, grad.w1, 48, 64);
		mulVecMat(grad.a03, w1, grad.a02, 48, 64);
		assignVecVec(grad.a03, grad.b1, 48);

		// relu[64]
		reluDerivativeVec(grad.a01, a01, grad.a02, 64);

		// dense[64x42]*[42]+[64]->[64]
		outerProduct(grad.a01, inp, grad.w0, 64, 42);
		mulVecMat(grad.a01, w0, grad.a00, 64, 42);
		assignVecVec(grad.a01, grad.b0, 64);
	}

	@Override
	public void learn(Gradient grad_, float negLearningRate) {
		CompiledGradient grad;
		try {
			grad = (CompiledGradient) grad_;
		} catch (ClassCastException e) {
			throw new IllegalArgumentException("" + //
					"grad_ must be of type " + //
					"de.a0h.minideeplearn.operation.compiler.output.CompiledChain.CompiledGradient, " + //
					"but it is a " + //
					grad_.getClass().getName(), e);
		}


		// dense[64x42]*[42]+[64]->[64]
		mulAddMatSca(grad.w0, negLearningRate, w0, 64, 42);
		mulAddVecSca(grad.b0, negLearningRate, b0, 64);

		// relu[64]

		// dense[48x64]*[64]+[48]->[48]
		mulAddMatSca(grad.w1, negLearningRate, w1, 48, 64);
		mulAddVecSca(grad.b1, negLearningRate, b1, 48);

		// relu[48]

		// dense[32x48]*[48]+[32]->[32]
		mulAddMatSca(grad.w2, negLearningRate, w2, 32, 48);
		mulAddVecSca(grad.b2, negLearningRate, b2, 32);

		// relu[32]

		// dense[24x32]*[32]+[24]->[24]
		mulAddMatSca(grad.w3, negLearningRate, w3, 24, 32);
		mulAddVecSca(grad.b3, negLearningRate, b3, 24);

		// relu[24]

		// dense[16x24]*[24]+[16]->[16]
		mulAddMatSca(grad.w4, negLearningRate, w4, 16, 24);
		mulAddVecSca(grad.b4, negLearningRate, b4, 16);

		// relu[16]

		// dense[7x16]*[16]+[7]->[7]
		mulAddMatSca(grad.w5, negLearningRate, w5, 7, 16);
		mulAddVecSca(grad.b5, negLearningRate, b5, 7);

		// softmax-ce[7]
	}

	@Override
	public String getTypeShortname() {
		return "compiled:chain";
	}

	@Override
	public String toString() {
		return toStringWithLayoutAndValues();
	}

	@Override
	public String toStringWithLayout() {
		return "" + //
				"compiled:" + //
				"chain(dense[64x42]*[42]+[64]->[64], " + //
				"relu[64], " + //
				"dense[48x64]*[64]+[48]->[48], " + //
				"relu[48], " + //
				"dense[32x48]*[48]+[32]->[32], " + //
				"relu[32], " + //
				"dense[24x32]*[32]+[24]->[24], " + //
				"relu[24], " + //
				"dense[16x24]*[24]+[16]->[16], " + //
				"relu[16], " + //
				"dense[7x16]*[16]+[7]->[7], " + //
				"softmax-ce[7])" //
		;
	}

	@Override
	public String toStringWithLayoutAndValues() {
		return toStringBuilderWithLayoutAndValues(new StringBuilder(), 0).toString();
	}

	@Override
	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
		return buf.append(toStringWithLayout());
	}

	protected static DecimalFormat createDecimalFormat() {
		DecimalFormatSymbols formatSymbols = DecimalFormatSymbols.getInstance(Locale.US);
		formatSymbols.setNaN("nan");
		formatSymbols.setInfinity("inf");

		DecimalFormat result = new DecimalFormat("0.########################");
		result.setDecimalFormatSymbols(formatSymbols);
		result.setDecimalSeparatorAlwaysShown(true);
		result.setGroupingUsed(false);

		return result;
	}

	private static final String SPACES = "                             ";

	@Override
	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {
		DecimalFormat format = createDecimalFormat();

		String indentSpaces = SPACES.substring(0, indent);
		buf.append(indentSpaces);
		toStringBuilderWithLayout(buf).append("\n");

		indent += 4;
		indentSpaces = SPACES.substring(0, indent);

		buf.append(indentSpaces).append("dense[64x42]*[42]+[64]->[64]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w0, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b0, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a01, buf, format).append("\n");

		buf.append(indentSpaces).append("relu[64]");
		buf.append(": out: ");
		toStringBuilderVec(a02, buf, format).append("\n");

		buf.append(indentSpaces).append("dense[48x64]*[64]+[48]->[48]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w1, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b1, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a03, buf, format).append("\n");

		buf.append(indentSpaces).append("relu[48]");
		buf.append(": out: ");
		toStringBuilderVec(a04, buf, format).append("\n");

		buf.append(indentSpaces).append("dense[32x48]*[48]+[32]->[32]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w2, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b2, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a05, buf, format).append("\n");

		buf.append(indentSpaces).append("relu[32]");
		buf.append(": out: ");
		toStringBuilderVec(a06, buf, format).append("\n");

		buf.append(indentSpaces).append("dense[24x32]*[32]+[24]->[24]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w3, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b3, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a07, buf, format).append("\n");

		buf.append(indentSpaces).append("relu[24]");
		buf.append(": out: ");
		toStringBuilderVec(a08, buf, format).append("\n");

		buf.append(indentSpaces).append("dense[16x24]*[24]+[16]->[16]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w4, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b4, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a09, buf, format).append("\n");

		buf.append(indentSpaces).append("relu[16]");
		buf.append(": out: ");
		toStringBuilderVec(a10, buf, format).append("\n");

		buf.append(indentSpaces).append("dense[7x16]*[16]+[7]->[7]");
		buf.append("\n");
		buf.append(indentSpaces).append("    weights: ");
		toStringBuilderMat(w5, buf, indent + 13, format).append("\n");
		buf.append(indentSpaces).append("    bias   : ");
		toStringBuilderVec(b5, buf, format).append("\n");
		buf.append(indentSpaces).append("    out    : ");
		toStringBuilderVec(a11, buf, format).append("\n");

		buf.append(indentSpaces).append("softmax-ce[7]");
		buf.append(": out: ");
		toStringBuilderVec(a12, buf, format).append("\n");
		buf.append(indentSpaces);
		buf.append("    loss: ");
		toStringBuilderSca(loss, buf, format);

		return buf;
	}

	public static class CompiledGradient implements Gradient {

		// input
		public final float[] a00 = new float[42];

		// dense[64x42]*[42]+[64]->[64]
		public final float[][] w0 = new float[64][42];
		public final float[] b0 = new float[64];
		public final float[] a01 = new float[64];

		// relu[64]
		public final float[] a02 = new float[64];

		// dense[48x64]*[64]+[48]->[48]
		public final float[][] w1 = new float[48][64];
		public final float[] b1 = new float[48];
		public final float[] a03 = new float[48];

		// relu[48]
		public final float[] a04 = new float[48];

		// dense[32x48]*[48]+[32]->[32]
		public final float[][] w2 = new float[32][48];
		public final float[] b2 = new float[32];
		public final float[] a05 = new float[32];

		// relu[32]
		public final float[] a06 = new float[32];

		// dense[24x32]*[32]+[24]->[24]
		public final float[][] w3 = new float[24][32];
		public final float[] b3 = new float[24];
		public final float[] a07 = new float[24];

		// relu[24]
		public final float[] a08 = new float[24];

		// dense[16x24]*[24]+[16]->[16]
		public final float[][] w4 = new float[16][24];
		public final float[] b4 = new float[16];
		public final float[] a09 = new float[16];

		// relu[16]
		public final float[] a10 = new float[16];

		// dense[7x16]*[16]+[7]->[7]
		public final float[][] w5 = new float[7][16];
		public final float[] b5 = new float[7];
		public final float[] a11 = new float[7];

		// softmax-ce[7]
		public final float[] a12 = new float[7];

		@Override
		public float[] getInputGrad() {
			return a00;
		}

		@Override
		public void clear() {
			// input
			assignVecSca(a00, 0.0f, 42);

			// dense[64x42]*[42]+[64]->[64]
			assignMatSca(w0, 0.0f, 64, 42);
			assignVecSca(b0, 0.0f, 64);
			assignVecSca(a01, 0.0f, 64);

			// relu[64]
			assignVecSca(a02, 0.0f, 64);

			// dense[48x64]*[64]+[48]->[48]
			assignMatSca(w1, 0.0f, 48, 64);
			assignVecSca(b1, 0.0f, 48);
			assignVecSca(a03, 0.0f, 48);

			// relu[48]
			assignVecSca(a04, 0.0f, 48);

			// dense[32x48]*[48]+[32]->[32]
			assignMatSca(w2, 0.0f, 32, 48);
			assignVecSca(b2, 0.0f, 32);
			assignVecSca(a05, 0.0f, 32);

			// relu[32]
			assignVecSca(a06, 0.0f, 32);

			// dense[24x32]*[32]+[24]->[24]
			assignMatSca(w3, 0.0f, 24, 32);
			assignVecSca(b3, 0.0f, 24);
			assignVecSca(a07, 0.0f, 24);

			// relu[24]
			assignVecSca(a08, 0.0f, 24);

			// dense[16x24]*[24]+[16]->[16]
			assignMatSca(w4, 0.0f, 16, 24);
			assignVecSca(b4, 0.0f, 16);
			assignVecSca(a09, 0.0f, 16);

			// relu[16]
			assignVecSca(a10, 0.0f, 16);

			// dense[7x16]*[16]+[7]->[7]
			assignMatSca(w5, 0.0f, 7, 16);
			assignVecSca(b5, 0.0f, 7);
			assignVecSca(a11, 0.0f, 7);

			// softmax-ce[7]
			assignVecSca(a12, 0.0f, 7);
		}

		@Override
		public void add(Gradient other) {
			CompiledGradient og;
			try {
				og = (CompiledGradient) other;
			} catch (ClassCastException e) {
				throw new IllegalArgumentException("" + //
						"other must be of type " + //
						"de.a0h.minideeplearn.operation.compiler.output.CompiledChain.CompiledGradient, " + //
						"but it is a " + //
						other.getClass().getName(), e);
			}

			// input
			addVec(a00, og.a00, 42);

			// dense[64x42]*[42]+[64]->[64]
			addMat(w0, og.w0, 64, 42);
			addVec(b0, og.b0, 64);
			addVec(a01, og.a01, 64);

			// relu[64]
			addVec(a02, og.a02, 64);

			// dense[48x64]*[64]+[48]->[48]
			addMat(w1, og.w1, 48, 64);
			addVec(b1, og.b1, 48);
			addVec(a03, og.a03, 48);

			// relu[48]
			addVec(a04, og.a04, 48);

			// dense[32x48]*[48]+[32]->[32]
			addMat(w2, og.w2, 32, 48);
			addVec(b2, og.b2, 32);
			addVec(a05, og.a05, 32);

			// relu[32]
			addVec(a06, og.a06, 32);

			// dense[24x32]*[32]+[24]->[24]
			addMat(w3, og.w3, 24, 32);
			addVec(b3, og.b3, 24);
			addVec(a07, og.a07, 24);

			// relu[24]
			addVec(a08, og.a08, 24);

			// dense[16x24]*[24]+[16]->[16]
			addMat(w4, og.w4, 16, 24);
			addVec(b4, og.b4, 16);
			addVec(a09, og.a09, 16);

			// relu[16]
			addVec(a10, og.a10, 16);

			// dense[7x16]*[16]+[7]->[7]
			addMat(w5, og.w5, 7, 16);
			addVec(b5, og.b5, 7);
			addVec(a11, og.a11, 7);

			// softmax-ce[7]
			addVec(a12, og.a12, 7);
		}

		@Override
		public void mul(float factor) {
			throw new UnsupportedOperationException("not yet implemented");
		}

		@Override
		public String getTypeShortname() {
			return "compiled_grad:chain";
		}

		@Override
		public String toString() {
			return toStringWithLayoutAndValues();
		}

		@Override
		public String toStringWithLayout() {
			return "" + //
					"compiled:" + //
					"chain_grad(dense_grad[64x42], " + //
					"vector_grad[64], " + //
					"dense_grad[48x64], " + //
					"vector_grad[48], " + //
					"dense_grad[32x48], " + //
					"vector_grad[32], " + //
					"dense_grad[24x32], " + //
					"vector_grad[24], " + //
					"dense_grad[16x24], " + //
					"vector_grad[16], " + //
					"dense_grad[7x16], " + //
					"vector_grad[7])" //
			;
		}

		@Override
		public String toStringWithLayoutAndValues() {
			return toStringBuilderWithLayoutAndValues(new StringBuilder(), 0).toString();
		}

		@Override
		public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {
			return buf.append(toStringWithLayout());
		}

		@Override
		public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {
			DecimalFormat format = createDecimalFormat();

			String indentSpaces = SPACES.substring(0, indent);
			buf.append(indentSpaces);
			toStringBuilderWithLayout(buf).append("\n");

			indent += 4;
			indentSpaces = SPACES.substring(0, indent);

			buf.append(indentSpaces).append("dense_grad[64x42]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a00, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w0, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b0, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[64]");
			buf.append(": inp: ");
			toStringBuilderVec(a01, buf, format).append("\n");

			buf.append(indentSpaces).append("dense_grad[48x64]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a02, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w1, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b1, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[48]");
			buf.append(": inp: ");
			toStringBuilderVec(a03, buf, format).append("\n");

			buf.append(indentSpaces).append("dense_grad[32x48]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a04, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w2, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b2, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[32]");
			buf.append(": inp: ");
			toStringBuilderVec(a05, buf, format).append("\n");

			buf.append(indentSpaces).append("dense_grad[24x32]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a06, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w3, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b3, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[24]");
			buf.append(": inp: ");
			toStringBuilderVec(a07, buf, format).append("\n");

			buf.append(indentSpaces).append("dense_grad[16x24]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a08, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w4, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b4, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[16]");
			buf.append(": inp: ");
			toStringBuilderVec(a09, buf, format).append("\n");

			buf.append(indentSpaces).append("dense_grad[7x16]");
			buf.append("\n");
			buf.append(indentSpaces).append("    inp    : ");
			toStringBuilderVec(a10, buf, format).append("\n");
			buf.append(indentSpaces).append("    weights: ");
			toStringBuilderMat(w5, buf, indent + 13, format).append("\n");
			buf.append(indentSpaces).append("    bias   : ");
			toStringBuilderVec(b5, buf, format).append("\n");

			buf.append(indentSpaces).append("vector_grad[7]");
			buf.append(": inp: ");
			toStringBuilderVec(a11, buf, format).append("\n");

			return buf;
		}
	}

	/**
	 * matA[i][j] += matB[i][j]
	 */
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
	private static void addVec(float[] vecA, float[] vecB, int length) {
		for (int i = 0; i < length; i++) {
			vecA[i] += vecB[i];
		}
	}

	/**
	 * Initializes the given matrix using the specified random number generator to
	 * generate a gaussian normal distribution.
	 * 
	 * @param stddev standard deviation of the distribution to use
	 */
	private static void assignGaussianMat(float[][] mat, float stddev, Random rnd, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] row = mat[i];

			for (int j = 0; j < colCount; j++) {
				row[j] = stddev * (float) rnd.nextGaussian();
			}
		}
	}

	/**
	 * Assign the specified value to every element of matrix mat.
	 */
	private static void assignMatSca(float[][] mat, float value, int rowCount, int colCount) {
		for (int i = 0; i < rowCount; i++) {
			float[] row = mat[i];
			for (int j = 0; j < colCount; j++) {
				row[j] = value;
			}
		}
	}

	/**
	 * Assign the specified value to every element of vector vec.
	 */
	private static void assignVecSca(float[] vec, float value, int length) {
		for (int i = 0; i < length; i++) {
			vec[i] = value;
		}
	}

	/**
	 * Assigns values of vecA to vecB, i.e. vecB[i] = vecA[i].
	 */
	private static void assignVecVec(float[] vecA, float[] vecB, int length) {
		for (int i = 0; i < length; i++) {
			vecB[i] = vecA[i];
		}
	}

	/**
	 * Multiplies matrix matA with scalar sca and then adds it to matrix matB,
	 * storing the result in matB. The matrix matA is not modified. I.e. matB[i][j]
	 * += sca * matA[i][j].
	 */
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
	 * Multiplies vector vecA with scalar sca and then adds it to vector vecB,
	 * storing the result in vecB. The vector vecA is not modified. I.e. vecB[i] +=
	 * sca * vecA[i].
	 */
	public static void mulAddVecSca(float[] vecA, float sca, float[] vecB, int length) {
		for (int j = 0; j < length; j++) {
			vecB[j] += sca * vecA[j];
		}
	}

	/**
	 * Multiplies matrix mat with vector inp, adds the bias vector and stores the
	 * result in out, i.e. calculates out = mat * inp + bias.
	 */
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

	/**
	 * Multiplies a row vector by a matrix and stores the result in an output
	 * vector, i.e. out = inp * mat.
	 */
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
	private static void outerProduct(float[] vec_u, float[] vec_v, float[][] mat, int uLength, int vLength) {
		for (int i = 0; i < uLength; i++) {
			float sca = vec_u[i];
			float[] row = mat[i];

			for (int j = 0; j < vLength; j++) {
				row[j] = sca * vec_v[j];
			}
		}
	}

	private static void reluDerivativeVec(float[] grad_inp, float[] inp, float[] grad_out, int length) {
		for (int i = 0; i < length; i++) {
			grad_inp[i] = (inp[i] >= 0) ? grad_out[i] : 0.0f;
		}
	}

	private static void reluVec(float[] inp, float[] out, int length) {
		for (int i = 0; i < length; i++) {
			out[i] = Math.max(0, inp[i]);
		}
	}

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

	protected static StringBuilder toStringBuilderSca(float sca, StringBuilder buf, DecimalFormat decimalFormat) {
		buf.append(decimalFormat.format(sca));

		return buf;
	}

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
}
