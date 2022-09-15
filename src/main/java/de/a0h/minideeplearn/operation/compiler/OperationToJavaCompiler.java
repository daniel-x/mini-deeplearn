package de.a0h.minideeplearn.operation.compiler;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Formatter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Consumer;

import de.a0h.javatemplater.JavaSource;
import de.a0h.javatemplater.MethodSourceTemplate;
import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.OperationUtil;
import de.a0h.minideeplearn.operation.activation.ActivationFunction;
import de.a0h.minideeplearn.operation.activation.Identity;
import de.a0h.minideeplearn.operation.activation.Relu;
import de.a0h.minideeplearn.operation.activation.Sigmoid;
import de.a0h.minideeplearn.operation.activation.Softmax;
import de.a0h.minideeplearn.operation.activation.Softplus;
import de.a0h.minideeplearn.operation.activation.Swish;
import de.a0h.minideeplearn.operation.activation.Tanh;
import de.a0h.minideeplearn.operation.composite.Chain;
import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.operation.layer.Dense;
import de.a0h.minideeplearn.operation.loss.CombinedLossFunction;
import de.a0h.minideeplearn.operation.loss.CrossEntropyLoss;
import de.a0h.minideeplearn.operation.loss.LossFunction;
import de.a0h.minideeplearn.operation.loss.SigmoidWithCrossEntropyLoss;
import de.a0h.minideeplearn.operation.loss.SoftmaxWithCrossEntropyLoss;

public class OperationToJavaCompiler {

	private static class Compilation {

		public Operation src;
		public Chain flattenedSrc;

		public boolean inlineLinalgOps;

		public StringBuilder outBuf;
		public Formatter out;

		public JavaSource dstInfo;

		public int storageVecCount = 0;
		public String storageVecFormat;
		public int storageVecIdx;

		public int paramMatCount = 0;
		public String paramMatFormat;
		public int paramMatIdx;

		public int paramVecCount = 0;
		public String paramVecFormat;
		public int paramVecIdx;

		public void resetVariableIndices() {
			storageVecIdx = 0;
			paramMatIdx = -1;
			paramVecIdx = -1;
		}

		HashMap<String, MethodSourceTemplate> methodTemplates = MethodTemplatesAccessible.getMethods();

		HashSet<String> methodDependencies = new HashSet<>();
	}

	public StringBuilder compile(Operation src) {
		return compile(src, false);
	}

	public StringBuilder compile(Operation src, boolean inlineVecOps) {
		String className = generateRandomClassName(src);

		return compile(src, new StringBuilder(8192), className, inlineVecOps);
	}

	public StringBuilder compile(Operation src, String dstClassName) {
		return compile(src, new StringBuilder(8192), dstClassName, false);
	}

	public StringBuilder compile(Operation src, String dstClassName, boolean inlineVecOps) {
		return compile(src, new StringBuilder(8192), dstClassName, inlineVecOps);
	}

	public StringBuilder compile(Operation src, StringBuilder dst, String dstClassName) {
		return compile(src, dst, dstClassName, false);
	}

	public StringBuilder compile(Operation src, StringBuilder dstBuf, String dstClassName, boolean inlineVecOps) {
		Compilation compi = new Compilation();

		compi.src = src;
		compi.flattenedSrc = flatten(src);

		compi.outBuf = dstBuf;
		compi.out = new Formatter(dstBuf);

		compi.dstInfo = new JavaSource();
		compi.dstInfo.setClassName(dstClassName);

		if (compi.dstInfo.packageName == null) {
			compi.dstInfo.packageName = getClass().getPackage().getName() + ".output";
		}

		compi.inlineLinalgOps = inlineVecOps;

		compi.dstInfo.importList.add(Arrays.class.getName());
		compi.dstInfo.importList.add(DecimalFormat.class.getName());
		compi.dstInfo.importList.add(DecimalFormatSymbols.class.getName());
		compi.dstInfo.importList.add(Locale.class.getName());
		compi.dstInfo.importList.add(Random.class.getName());
		compi.dstInfo.importList.add(null);
		compi.dstInfo.importList.add(Operation.class.getName());
		compi.dstInfo.importList.add(Gradient.class.getName());
		if (!src.hasLoss() || !src.hasOutput()) {
			compi.dstInfo.importList.add(OperationUtil.class.getName());
		}

		compi.dstInfo.caption = String.format( //
				"public class %s implements %s {\n", //
				compi.dstInfo.simpleClassName, Operation.class.getSimpleName());

		initVariableCounts(compi);

		int maxDigits;
		maxDigits = Integer.toString(compi.storageVecCount).length();
		compi.storageVecFormat = "a%0" + maxDigits + "d";
		maxDigits = Integer.toString(compi.paramMatCount - 1).length();
		compi.paramMatFormat = "w%0" + maxDigits + "d";
		maxDigits = Integer.toString(compi.paramVecCount - 1).length();
		compi.paramVecFormat = "b%0" + maxDigits + "d";

		compileJavaFile(compi);

		compi.out.close();

		return dstBuf;
	}

	public static void upChar(StringBuilder buf, int idx) {
		char c = buf.charAt(idx);
		c = Character.toUpperCase(c);
		buf.setCharAt(idx, c);
	}

	public static String upChar(String s, int idx) {
		char c = s.charAt(idx);
		c = Character.toUpperCase(c);

		return s.substring(0, idx) + c + s.substring(idx + 1);
	}

	public static String generateRandomClassName(Operation op) {
		StringBuilder result = new StringBuilder();
		result.append("CompiledOperation");

		int upCharIdx = result.length();
		result.append(op.getTypeShortname());
		upChar(result, upCharIdx);

		result.append((int) (Math.random() * 1000000000));

		return result.toString();
	}

	private Chain flatten(Operation src) {
		if (src instanceof Chain) {
			return ((Chain) src).flattened();
		} else {
			Chain result = new Chain();
			result.add(src);
			return result;
		}
	}

	private void initVariableCounts(Compilation compi) {
		for (Operation op : compi.flattenedSrc) {
			if (op instanceof Dense) {
				compi.storageVecCount++;
				compi.paramMatCount++;
				compi.paramVecCount++;

			} else if (op instanceof Swish) {
				compi.storageVecCount += 2;

			} else if (op instanceof CombinedLossFunction) {
				compi.storageVecCount++;

			} else if (op instanceof ActivationFunction) {
				compi.storageVecCount++;

			} else if (op instanceof LossFunction) {
				// no additional variables needed

			} else {
				throw new UnsupportedOperationException("" + //
						"unknown operation type: " + op.getTypeShortname() + //
						" (" + op.getClass().getName() + ")");
			}
		}
	}

	private void compileJavaFile(Compilation compi) {
		generateJavaFileLeadingCode(compi);

		compileVariableDeclarations(compi);

		compileInitParams(compi);
		compileGetInputSize(compi);
		compileHasOutput(compi);
		compileGetOutputSize(compi);
		compileCalcOutput(compi);
		compileGetOutput(compi);
		compileHasLoss(compi);
		compileCalcLoss(compi);
		compileGetLoss(compi);
		compileCreateGradient(compi);
		compileCalcGradient(compi);
		compileLearn(compi);
		compileGetTypeShortname(compi);
		compileToString(compi);
		compileToStringWithLayout(compi);
		compileToStringWithLayoutAndValues(compi);
		compileToStringBuilderWithLayout(compi);
		compileCreateDecimalFormat(compi);
		compileToStringBuilderWithLayoutAndValues(compi);

		compileGradientClass(compi);

		compileRequiredMethods(compi);

		compi.out.format("}\n");
	}

	private static void generateJavaFileLeadingCode(Compilation compi) {
		compi.out.flush();
		StringBuilder buf = compi.outBuf;

		String packageName = compi.dstInfo.packageName;
		String caption = compi.dstInfo.caption;
		List<String> importList = compi.dstInfo.importList;

		if (packageName != null) {
			buf.append("package ").append(packageName).append(";\n");
		}

		if (importList != null) {
			ifNotEmptyAppendLinefeed(buf);
			for (String imported : importList) {
				if (imported != null) {
					buf.append("import ").append(imported).append(";\n");
				} else {
					buf.append("\n");
				}
			}
			ifNotEmptyAppendLinefeed(buf);
		}

		if (caption != null) {
			buf.append(caption);
		}
	}

	protected static void ifNotEmptyAppendLinefeed(StringBuilder buf) {
		if (buf.length() > 0) {
			buf.append("\n");
		}
	}

	private void compileGetInputSize(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public int getInputSize() {\n");
		compi.out.format("		return %d;\n", compi.src.getInputSize());
		compi.out.format("	}\n");
	}

	private void compileHasOutput(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public boolean hasOutput() {\n");
		compi.out.format("		return %s;\n", compi.src.hasOutput());
		compi.out.format("	}\n");
	}

	private void compileGetOutputSize(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public int getOutputSize() {\n");

		if (compi.src.hasOutput()) {
			compi.out.format("		return %d;\n", compi.src.getOutputSize());
		} else {
			compi.out.format("		throwOutputNotProvidedException(\"%s\");\n", compi.src.getClass().getName());
			compi.out.format("		return -1;\n");

			compi.methodDependencies.add("throwOutputNotProvidedException");
		}

		compi.out.format("	}\n");
	}

	private void compileGetOutput(Compilation compi) {
		String outName = String.format(compi.storageVecFormat, compi.storageVecCount);

		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public float[] getOutput() {\n");

		if (compi.src.hasOutput()) {
			compi.out.format("		return %s;\n", outName);
		} else {
			compi.out.format("		throwOutputNotProvidedException(\"%s\");\n", compi.src.getClass().getName());
			compi.out.format("		return null;\n");

			compi.methodDependencies.add("throwOutputNotProvidedException");
		}

		compi.out.format("	}\n");
	}

	private void compileHasLoss(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public boolean hasLoss() {\n");
		compi.out.format("		return %s;\n", compi.src.hasLoss());
		compi.out.format("	}\n");
	}

	private void compileCalcLoss(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public float calcLoss(float[] inp, float[] target) {\n");

		if (compi.src.hasLoss()) {
			Operation lastOp = compi.flattenedSrc.getLast();

			if (false || //
					lastOp instanceof CrossEntropyLoss || //
					lastOp instanceof SigmoidWithCrossEntropyLoss || //
					lastOp instanceof SoftmaxWithCrossEntropyLoss //
			) {
				String predictedName = String.format(compi.storageVecFormat, compi.storageVecCount);
				compileCrossEntropyLoss(compi, predictedName, "target", "loss");
				compi.out.format("\n");
				compi.out.format("		return loss;\n");

			} else {
				throwUnsupportedOperationException(lastOp);

			}

		} else {
			compi.out.format("		throwLossNotProvidedException(\"%s\");\n", compi.src.getClass().getName());
			compi.out.format("		return -1.0f;\n");

			compi.methodDependencies.add("throwLossNotProvidedException");
		}

		compi.out.format("	}\n");
	}

	private void compileGetLoss(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public float getLoss() {\n");

		if (compi.src.hasLoss()) {
			compi.out.format("		return loss;\n");
		} else {
			compi.out.format("		throwLossNotProvidedException(\"%s\");\n", compi.src.getClass().getName());
			compi.out.format("		return -1.0f;\n");

			compi.methodDependencies.add("throwLossNotProvidedException");
		}

		compi.out.format("	}\n");
	}

	private void compileCalcGradient(Compilation compi) {
		String target_or_upstream_grad_of_out_name;
		if (compi.src.hasLoss()) {
			target_or_upstream_grad_of_out_name = "target";
		} else {
			target_or_upstream_grad_of_out_name = "grad_backprop";
		}

		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("	@Override\n");
		aut.format("	public void calcGradient( //\n");
		aut.format("			float[] inp, //\n");
		aut.format("			float[] %s, //\n", target_or_upstream_grad_of_out_name);
		aut.format("			Gradient grad_ //\n");
		aut.format("	) {\n");
		aut.format("		CompiledGradient grad;\n");
		aut.format("		try {\n");
		aut.format("			grad = (CompiledGradient) grad_;\n");
		aut.format("		} catch (ClassCastException e) {\n");
		aut.format("			throw new IllegalArgumentException(\"\" + //\n");
		aut.format("					\"grad_ must be of type \" + //\n");
		aut.format("					\"%s.CompiledGradient, \" + //\n", compi.dstInfo.getClassName());
		aut.format("					\"but it is a \" + //\n");
		aut.format("					grad_.getClass().getName(), e);\n");
		aut.format("		}\n");

		compi.storageVecIdx = compi.storageVecCount;
		compi.paramMatIdx = compi.paramMatCount - 1;
		compi.paramVecIdx = compi.paramVecCount - 1;

		Chain flattenedSrc = compi.flattenedSrc;

		int i = flattenedSrc.size() - 1;
		if (compi.src.hasLoss()) {
			Operation op = flattenedSrc.get(i--);

			aut.format("\n");
			aut.format("		// %s\n", op.toStringWithLayout());

			String outName = String.format(compi.storageVecFormat, compi.storageVecIdx--);
			String inpName = String.format(compi.storageVecFormat, compi.storageVecIdx);
			String inpSize = Integer.toString(op.getInputSize());

			if (op instanceof SigmoidWithCrossEntropyLoss) {
				if (op.getInputSize() == 1) {
					compileTemplate(compi, "", "sigmoidWithCrossEntropyLossGradient_twofoldClassification", outName,
							"target", "grad." + inpName);
				} else {
					compileTemplate(compi, "", "sigmoidWithCrossEntropyLossGradient_manifoldClassification", outName,
							"target", inpSize, "grad." + inpName);
				}

			} else if (op instanceof SoftmaxWithCrossEntropyLoss) {
				compileTemplate(compi, "", "softmaxWithCrossEntropyLossGradient_manifoldClassification", outName,
						"target", inpSize, "grad." + inpName);

			} else if (op instanceof CrossEntropyLoss) {
				if (op.getInputSize() == 1) {
					compileTemplate(compi, "", "crossEntropyLossGradient_twofoldClassification", outName, "target",
							"grad." + outName);
				} else {
					compileTemplate(compi, "", "crossEntropyLossGradient_manifoldClassification", outName, "target",
							inpSize, "grad." + outName);
				}

			} else {
				throwUnsupportedOperationException(op);
			}
		}

		for (; i >= 0; i--) {
			Operation op = flattenedSrc.get(i);

			aut.format("\n");
			aut.format("		// %s\n", op.toStringWithLayout());

			String out = String.format(compi.storageVecFormat, compi.storageVecIdx);
			String inp = String.format(compi.storageVecFormat, --compi.storageVecIdx);
			String gradOut = "grad." + out;
			String gradInp = "grad." + inp;
			String outLength = Integer.toString(op.getOutputSize());
			String inpLength = Integer.toString(op.getInputSize());
			if (compi.storageVecIdx == 0) {
				inp = "inp";
			}

			if (op instanceof Dense) {
				String mat = String.format(compi.paramMatFormat, compi.paramMatIdx--);
				String gradMat = "grad." + mat;
				String gradBias = "grad." + String.format(compi.paramVecFormat, compi.paramVecIdx--);
				compileTemplate(compi, "", "outerProduct", gradOut, inp, gradMat, outLength, inpLength);
				compileTemplate(compi, "", "mulVecMat", gradOut, mat, gradInp, outLength, inpLength);
				compileTemplate(compi, "", "assignVecVec", gradOut, gradBias, outLength);

			} else if (op instanceof Tanh) {
				compileTemplate(compi, "", "tanhDerivativeVec", gradInp, out, gradOut, inpLength);

			} else if (op instanceof Sigmoid) {
				compileTemplate(compi, "", "sigmoidDerivativeVec", gradInp, out, gradOut, inpLength);

			} else if (op instanceof Softplus) {
				compileTemplate(compi, "", "softplusDerivativeVec", gradInp, inp, gradOut, inpLength);

			} else if (op instanceof Relu) {
				compileTemplate(compi, "", "reluDerivativeVec", gradInp, inp, gradOut, inpLength);

			} else if (op instanceof Identity) {
				compileTemplate(compi, "", "assignVecVec", gradOut, gradInp, inpLength);

			} else if (op instanceof Swish) {
				String sig = inp;
				inp = String.format(compi.storageVecFormat, --compi.storageVecIdx);
				gradInp = "grad." + inp;
				compileTemplate(compi, "", "swishDerivativeVec", gradInp, out, sig, gradOut, inpLength);

			} else if (op instanceof Softmax) {
				compileTemplate(compi, "", "softmaxDerivativeVec", gradInp, inp, gradOut, inpLength);

			} else if (op instanceof CombinedLossFunction) {
				throw new UnsupportedOperationException("not yet implemented: " + op.getClass().getName());

			} else if (op instanceof ActivationFunction) {
				throw new UnsupportedOperationException("not yet implemented: " + op.getClass().getName());

			} else if (op instanceof LossFunction) {
				// nothing to do for pure loss function

			} else {
				throwUnsupportedOperationException(op);
			}
		}

		aut.format("	}\n");
	}

	private void compileInitParams(Compilation compi) {
		Formatter out = compi.out;

		String xavierFactor = "xavierFactor";
		String rnd = "rnd";

		out.format("\n");
		out.format("	@Override\n");
		out.format("	public void initParams(Random %s) {\n", rnd);

		out.format("		float %s;\n\n", xavierFactor);

		compi.resetVariableIndices();
		for (Operation op : compi.flattenedSrc) {
			out.format("		// %s\n", op.toStringWithLayout());

			if (op instanceof Dense) {
				String inpSize = Integer.toString(op.getInputSize());
				String outSize = Integer.toString(op.getOutputSize());
				String mat = String.format(compi.paramMatFormat, ++compi.paramMatIdx);
				String bias = String.format(compi.paramVecFormat, ++compi.paramVecIdx);
				out.format("		%s = 2.0f / (%s + %s);\n", xavierFactor, inpSize, outSize);
				compileTemplate(compi, "", "assignGaussianMat", mat, xavierFactor, rnd, outSize, inpSize);
				compileTemplate(compi, "", "assignVecSca", bias, "0.0f", outSize);

			} else if (op instanceof CombinedLossFunction) {
			} else if (op instanceof ActivationFunction) {
			} else if (op instanceof LossFunction) {
			} else {
				throwUnsupportedOperationException(op);
			}

			out.format("\n");
		}

		out.format("	}\n");
	}

	private void compileCreateGradient(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public CompiledGradient createGradient() {\n");
		compi.out.format("		return new CompiledGradient();\n");
		compi.out.format("	}\n");
	}

	private void compileGetTypeShortname(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public String getTypeShortname() {\n");
		compi.out.format("		return \"compiled:%s\";\n", compi.src.getTypeShortname());
		compi.out.format("	}\n");
	}

	private void compileToString(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public String toString() {\n");
		compi.out.format("		return toStringWithLayoutAndValues();\n");
		compi.out.format("	}\n");
	}

	private void compileToStringWithLayout(Compilation compi) {
		String str = compi.src.toStringWithLayout();
		str = str.replace(" ", " \" + //\n				\"");

		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public String toStringWithLayout() {\n");
		compi.out.format("		return \"\" + //\n");
		compi.out.format("				\"compiled:\" + //\n");
		compi.out.format("				\"%s\" //\n", str);
		compi.out.format("		;\n");
		compi.out.format("	}\n");
	}

	private void compileToStringBuilderWithLayout(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {\n");
		compi.out.format("		return buf.append(toStringWithLayout());\n");
		compi.out.format("	}\n");
	}

	private void compileToStringWithLayoutAndValues(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("	@Override\n");
		aut.format("	public String toStringWithLayoutAndValues() {\n");
		aut.format("		return toStringBuilderWithLayoutAndValues(new StringBuilder(), 0).toString();\n");
		aut.format("	}\n");
	}

	private void compileCreateDecimalFormat(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("	protected static DecimalFormat createDecimalFormat() {\n");
		aut.format("		DecimalFormatSymbols formatSymbols = DecimalFormatSymbols.getInstance(Locale.US);\n");
		aut.format("		formatSymbols.setNaN(\"nan\");\n");
		aut.format("		formatSymbols.setInfinity(\"inf\");\n");
		aut.format("\n");
		aut.format("		DecimalFormat result = new DecimalFormat(\"0.########################\");\n");
		aut.format("		result.setDecimalFormatSymbols(formatSymbols);\n");
		aut.format("		result.setDecimalSeparatorAlwaysShown(true);\n");
		aut.format("		result.setGroupingUsed(false);\n");
		aut.format("\n");
		aut.format("		return result;\n");
		aut.format("	}\n");
	}

	private void compileToStringBuilderWithLayoutAndValues(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("	private static final String SPACES = \"                             \";\n");
		aut.format("\n");
		aut.format("	@Override\n");
		aut.format("	public StringBuilder toStringBuilderWithLayoutAndValues(StringBuilder buf, int indent) {\n");
		aut.format("		DecimalFormat format = createDecimalFormat();\n");
		aut.format("\n");
		aut.format("		String indentSpaces = SPACES.substring(0, indent);\n");
		aut.format("		buf.append(indentSpaces);\n");
		aut.format("		toStringBuilderWithLayout(buf).append(\"\\n\");\n");
		aut.format("\n");
		aut.format("		indent += 4;\n");
		aut.format("		indentSpaces = SPACES.substring(0, indent);\n");

		String storageVecFormat = "		toStringBuilderVec(" + //
				compi.storageVecFormat + ", buf, format).append(\"\\n\");\n";
		String paramMatFormat = "		toStringBuilderMat(" + //
				compi.paramMatFormat + ", buf, indent + 13, format).append(\"\\n\");\n";
		String paramVecFormat = "		toStringBuilderVec(" + //
				compi.paramVecFormat + ", buf, format).append(\"\\n\");\n";

		compi.resetVariableIndices();
		for (Operation op : compi.flattenedSrc) {
			aut.format("\n");
			aut.format("		buf.append(indentSpaces).append(\"" + op.toStringWithLayout() + "\");\n");

			if (op instanceof Dense) {
				aut.format("		buf.append(\"\\n\");\n");
				aut.format("		buf.append(indentSpaces).append(\"    weights: \");\n");
				aut.format(paramMatFormat, ++compi.paramMatIdx);

				aut.format("		buf.append(indentSpaces).append(\"    bias   : \");\n");
				aut.format(paramVecFormat, ++compi.paramVecIdx);

				aut.format("		buf.append(indentSpaces).append(\"    out    : \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);

				compi.methodDependencies.add("toStringBuilderMat");
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof Swish) {
				++compi.storageVecIdx;
				aut.format("		buf.append(\": out: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof CombinedLossFunction) {
				aut.format("		buf.append(\": out: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				aut.format("		buf.append(indentSpaces);\n");
				aut.format("		buf.append(\"    loss: \");\n");
				aut.format("		toStringBuilderSca(loss, buf, format);\n");
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof ActivationFunction) {
				aut.format("		buf.append(\": out: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof LossFunction) {
				aut.format("		buf.append(\": loss: \");\n");
				aut.format("		toStringBuilderSca(loss, buf, format);\n");
				compi.methodDependencies.add("toStringBuilderSca");

			} else {
				throwUnsupportedOperationException(op);
			}
		}

		aut.format("\n");
		aut.format("		return buf;\n");
		aut.format("	}\n");
	}

	private void compileLearn(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("	@Override\n");
		aut.format("	public void learn(Gradient grad_, float negLearningRate) {\n");
		aut.format("		CompiledGradient grad;\n");
		aut.format("		try {\n");
		aut.format("			grad = (CompiledGradient) grad_;\n");
		aut.format("		} catch (ClassCastException e) {\n");
		aut.format("			throw new IllegalArgumentException(\"\" + //\n");
		aut.format("					\"grad_ must be of type \" + //\n");
		aut.format("					\"%s.CompiledGradient, \" + //\n", compi.dstInfo.getClassName());
		aut.format("					\"but it is a \" + //\n");
		aut.format("					grad_.getClass().getName(), e);\n");
		aut.format("		}\n");

		Consumer<int[]> storageVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
			}
		};

		Consumer<int[]> paramMatCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String outSize = "" + args[1];
				String inpSize = "" + args[2];
				String varName = String.format(compi.paramMatFormat, idx);
				compileTemplate(compi, "", "mulAddMatSca", "grad." + varName, "negLearningRate", varName, outSize,
						inpSize);
			}
		};

		Consumer<int[]> paramVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String inpSize = "" + args[1];
				String varName = String.format(compi.paramVecFormat, idx);
				compileTemplate(compi, "", "mulAddVecSca", "grad." + varName, "negLearningRate", varName, inpSize);
			}
		};

		Consumer<String> commentCodeGenerator = new Consumer<String>() {
			public void accept(String comment) {
				aut.format("\t\t// %s\n", comment);
			}
		};

		compi.resetVariableIndices();

		aut.format("\n");

		compileVariableHandling( //
				compi, //
				storageVecCodeGenerator, //
				paramMatCodeGenerator, //
				paramVecCodeGenerator, //
				commentCodeGenerator //
		);

		aut.format("	}\n");
	}

	private void compileCrossEntropyLoss(Compilation compi, String predictedName, String targetName, String lossName) {
		int outSize = compi.src.getOutputSize();

		if (outSize == 1) {
			compileTemplate(compi, "", "crossEntropyLoss_twofoldClassification", //
					predictedName, targetName, lossName);

		} else {
			String lengthName = Integer.toString(outSize);

			compileTemplate(compi, "", "crossEntropyLoss_manifoldClassification", //
					predictedName, targetName, lengthName, lossName);

		}
	}

	private void compileVariableDeclarations(Compilation compi) {
		compi.resetVariableIndices();

		String commentFormat = "	// %s\n";
		String paramMatFormat = "	public final float[][] " + compi.paramMatFormat + " = new float[%d][%d];\n";
		String paramVecFormat = "	public final float[] " + compi.paramVecFormat + " = new float[%d];\n";
		String storageVecFormat = "	public final float[] " + compi.storageVecFormat + " = new float[%d];\n";

		compileVariableHandling( //
				compi, //
				storageVecFormat, //
				paramMatFormat, //
				paramVecFormat, //
				commentFormat //
		);

		if (compi.src.hasLoss()) {
			compi.out.format("	public float loss;\n");
		}
	}

	private void compileVariableHandling( //
			Compilation compi, //
			String storageVecFormat, //
			String paramMatFormat, //
			String paramVecFormat, //
			String commentFormat //
	) {
		for (Operation op : compi.flattenedSrc) {
			compi.out.format("\n");
			compi.out.format(commentFormat, op.toStringWithLayout());

			int inpSize = op.getInputSize();

			if (op instanceof Dense) {
				int outSize = op.getOutputSize();
				compi.out.format(paramMatFormat, ++compi.paramMatIdx, outSize, inpSize);
				compi.out.format(paramVecFormat, ++compi.paramVecIdx, outSize);
				compi.out.format(storageVecFormat, ++compi.storageVecIdx, outSize);

			} else if (op instanceof Swish) {
				compi.out.format(storageVecFormat, ++compi.storageVecIdx, inpSize);
				compi.out.format(storageVecFormat, ++compi.storageVecIdx, inpSize);

			} else if (op instanceof CombinedLossFunction) {
				compi.out.format(storageVecFormat, ++compi.storageVecIdx, inpSize);

			} else if (op instanceof ActivationFunction) {
				compi.out.format(storageVecFormat, ++compi.storageVecIdx, inpSize);

			} else if (op instanceof LossFunction) {
				// nothing to do for pure loss function

			} else {
				throwUnsupportedOperationException(op);
			}
		}
	}

	private void compileVariableHandling(Compilation compi, //
			Consumer<int[]> storageVecCodeGenerator, //
			Consumer<int[]> paramMatCodeGenerator, //
			Consumer<int[]> paramVecCodeGenerator, //
			Consumer<String> commentCodeGenerator //
	) {
		for (Operation op : compi.flattenedSrc) {
			compi.out.format("\n");
			commentCodeGenerator.accept(op.toStringWithLayout());

			int inpSize = op.getInputSize();

			if (op instanceof Dense) {
				int outSize = op.getOutputSize();
				paramMatCodeGenerator.accept(new int[] { ++compi.paramMatIdx, outSize, inpSize });
				paramVecCodeGenerator.accept(new int[] { ++compi.paramVecIdx, outSize });
				storageVecCodeGenerator.accept(new int[] { ++compi.storageVecIdx, outSize });

			} else if (op instanceof Swish) {
				storageVecCodeGenerator.accept(new int[] { ++compi.storageVecIdx, inpSize });
				storageVecCodeGenerator.accept(new int[] { ++compi.storageVecIdx, inpSize });

			} else if (op instanceof CombinedLossFunction) {
				storageVecCodeGenerator.accept(new int[] { ++compi.storageVecIdx, inpSize });

			} else if (op instanceof ActivationFunction) {
				storageVecCodeGenerator.accept(new int[] { ++compi.storageVecIdx, inpSize });

			} else if (op instanceof LossFunction) {
				// nothing to do for pure loss function

			} else {
				throwUnsupportedOperationException(op);
			}
		}
	}

	private void compileCalcOutput(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	@Override\n");
		compi.out.format("	public float[] calcOutput(float[] " + compi.storageVecFormat + ") {\n", 0);

		compileInlineCalcOutput(compi);

		compi.out.format("		return " + compi.storageVecFormat + ";\n", compi.storageVecIdx);
		compi.out.format("	}\n");
	}

	private void compileInlineCalcOutput(Compilation compi) {
		compi.resetVariableIndices();

		for (Operation op : compi.flattenedSrc) {
			compi.out.format("		// %s\n", op.toStringWithLayout());

			String inpName = String.format(compi.storageVecFormat, compi.storageVecIdx);
			String inpSize = Integer.toString(op.getInputSize());

			if (op instanceof Dense) {
				String outName = String.format(compi.storageVecFormat, ++compi.storageVecIdx);
				String matName = String.format(compi.paramMatFormat, ++compi.paramMatIdx);
				String biasName = String.format(compi.paramVecFormat, ++compi.paramVecIdx);
				String outSize = Integer.toString(op.getOutputSize());
				compileTemplate(compi, "", "mulMatVecPlusBias", //
						matName, inpName, biasName, outName, outSize, inpSize);

			} else if (op instanceof ActivationFunction) {
				String outName = String.format(compi.storageVecFormat, ++compi.storageVecIdx);

				if (op instanceof Tanh) {
					compileTemplate(compi, "", "tanhVec", inpName, outName, inpSize);

				} else if (op instanceof SigmoidWithCrossEntropyLoss || op instanceof Sigmoid) {
					compileTemplate(compi, "", "sigmoidVec", inpName, outName, inpSize);

				} else if (op instanceof SoftmaxWithCrossEntropyLoss || op instanceof Softmax) {
					compileTemplate(compi, "", "softmaxVec", inpName, outName, inpSize);

				} else if (op instanceof Softplus) {
					compileTemplate(compi, "", "softplusVec", inpName, outName, inpSize);

				} else if (op instanceof Relu) {
					compileTemplate(compi, "", "reluVec", inpName, outName, inpSize);

				} else if (op instanceof Identity) {
					compileTemplate(compi, "", "assignVecVec", inpName, outName, inpSize);

				} else if (op instanceof Swish) {
					String sigName = outName;
					outName = String.format(compi.storageVecFormat, ++compi.storageVecIdx);
					compileTemplate(compi, "", "swishVec", inpName, sigName, outName, inpSize);

				} else {
					throwUnsupportedOperationException(op);
				}

			} else if (op instanceof CrossEntropyLoss) {
				// no output calculated for a pure loss operation

			} else {
				throwUnsupportedOperationException(op);
			}

			compi.out.format("\n");
		}
	}

	private void compileTemplate(Compilation compi, String indentPrefix, String templateName,
			String... paramExpressionList) {
		MethodSourceTemplate template = compi.methodTemplates.get(templateName);

		if (template == null) {
			throw new IllegalArgumentException("unknown template name: " + templateName);
		}

		if (paramExpressionList.length != template.paramList.size()) {
			throwIllegalArgumentExceptionParamListsDiffer(template, paramExpressionList);
		}

		if (compi.inlineLinalgOps || template.mustInline) {
			String[] zippedReplaceMap = new String[2 * (paramExpressionList.length + 1)];
			for (int i = 0; i < paramExpressionList.length; i++) {
				zippedReplaceMap[2 * i + 0] = template.paramList.get(i).name;
				zippedReplaceMap[2 * i + 1] = paramExpressionList[i];
			}
			zippedReplaceMap[zippedReplaceMap.length - 2] = "\n";
			zippedReplaceMap[zippedReplaceMap.length - 1] = "\n" + indentPrefix;

			String body = template.body;
			if (!body.startsWith("\n")) {
				body = indentPrefix + body;
			}
			String code = replaceMany(body, zippedReplaceMap);
			if (code.endsWith(indentPrefix)) {
				code = code.substring(0, code.length() - indentPrefix.length());
			}

			compi.out.format(code);
		} else {
			compi.out.format(indentPrefix + "\t\t%s(", templateName);
			for (int i = 0; i < paramExpressionList.length; i++) {
				compi.out.format(paramExpressionList[i]);

				if (i < template.paramList.size() - 1) {
					compi.out.format(", ");
				}
			}
			compi.out.format(");\n");

			compi.methodDependencies.add(templateName);
		}
	}

	protected void throwIllegalArgumentExceptionParamListsDiffer(MethodSourceTemplate template,
			String... paramExpressionList) {
		String[] expectedParamList = new String[template.paramList.size()];
		for (int i = 0; i < template.paramList.size(); i++) {
			expectedParamList[i] = template.paramList.get(i).name;
		}
		String expectedParams = String.join(", ", expectedParamList);
		String actualParams = Arrays.toString(paramExpressionList);

		throw new IllegalArgumentException(String.format( //
				"template \"%s\" requires %d parameters [%s], but specified expression list has %d %s", //
				template.name, //
				template.paramList.size(), expectedParams, //
				paramExpressionList.length, actualParams));
	}

	private void compileRequiredMethods(Compilation compi) {
//		Class<? extends OperationToJavaCompiler> thisClass = getClass();
//		Class<?>[] paramTypes = { Compilation.class };
		compi.out.flush();
		StringBuilder buf = compi.outBuf;

		ArrayList<String> sortedMethodDependencies = new ArrayList<>(compi.methodDependencies);
		Collections.sort(sortedMethodDependencies);

		for (String dependency : sortedMethodDependencies) {
			MethodSourceTemplate method = compi.methodTemplates.get(dependency);

			if (method == null) {
				throw new RuntimeException("couldn't get source of method " + dependency);
			}

			method.generateSource(buf);
		}
	}

	private void compileGradientClass(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("	public static class CompiledGradient implements %s {\n", Gradient.class.getSimpleName());

		compileGradientVariableDeclarations(compi);

		compileGradientGetInputGrad(compi);
		compileGradientClear(compi);
		compileGradientAdd(compi);
		compileGradientMul(compi);
		compileGradientGetTypeShortname(compi);
		compileGradientToString(compi);
		compileGradientToStringWithLayout(compi);
		compileGradientToStringWithLayoutAndValues(compi);
		compileGradientToStringBuilderWithLayout(compi);
		compileGradientToStringBuilderWithLayoutAndValues(compi);

		compi.out.format("	}\n");
	}

	private void compileGradientGetInputGrad(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public float[] getInputGrad() {\n");
		compi.out.format("			return " + compi.storageVecFormat + ";\n", 0);
		compi.out.format("		}\n");
	}

	private void compileGradientAdd(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("		@Override\n");
		aut.format("		public void add(Gradient other) {\n");
		aut.format("			CompiledGradient og;\n");
		aut.format("			try {\n");
		aut.format("				og = (CompiledGradient) other;\n");
		aut.format("			} catch (ClassCastException e) {\n");
		aut.format("				throw new IllegalArgumentException(\"\" + //\n");
		aut.format("						\"other must be of type \" + //\n");
		aut.format("						\"%s.CompiledGradient, \" + //\n", compi.dstInfo.getClassName());
		aut.format("						\"but it is a \" + //\n");
		aut.format("						other.getClass().getName(), e);\n");
		aut.format("			}\n");
		aut.format("\n");

		Consumer<int[]> storageVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String inpSize = "" + args[1];
				String varName = String.format(compi.storageVecFormat, idx);
				compileTemplate(compi, "\t", "addVec", varName, "og." + varName, inpSize);
			}
		};

		Consumer<int[]> paramMatCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String outSize = "" + args[1];
				String inpSize = "" + args[2];
				String varName = String.format(compi.paramMatFormat, idx);
				compileTemplate(compi, "\t", "addMat", varName, "og." + varName, outSize, inpSize);
			}
		};

		Consumer<int[]> paramVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String inpSize = "" + args[1];
				String varName = String.format(compi.paramVecFormat, idx);
				compileTemplate(compi, "\t", "addVec", varName, "og." + varName, inpSize);
			}
		};

		Consumer<String> commentCodeGenerator = new Consumer<String>() {
			public void accept(String comment) {
				aut.format("\t\t\t// %s\n", comment);
			}
		};

		compi.resetVariableIndices();

		commentCodeGenerator.accept("input");
		storageVecCodeGenerator.accept(new int[] { compi.storageVecIdx, compi.src.getInputSize() });

		compileVariableHandling( //
				compi, //
				storageVecCodeGenerator, //
				paramMatCodeGenerator, //
				paramVecCodeGenerator, //
				commentCodeGenerator //
		);

		aut.format("		}\n");
	}

	private void compileGradientMul(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public void mul(float factor) {\n");
		compi.out.format("			throw new UnsupportedOperationException(\"not yet implemented\");\n");
		compi.out.format("		}\n");
	}

	private void compileGradientGetTypeShortname(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public String getTypeShortname() {\n");
		compi.out.format("			return \"compiled_grad:%s\";\n", compi.src.getTypeShortname());
		compi.out.format("		}\n");
	}

	private void compileGradientToString(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public String toString() {\n");
		compi.out.format("			return toStringWithLayoutAndValues();\n");
		compi.out.format("		}\n");
	}

	private void compileGradientToStringWithLayout(Compilation compi) {
		String str = compi.src.createGradient().toStringWithLayout();
		str = str.replace(" ", " \" + //\n					\"");

		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public String toStringWithLayout() {\n");
		compi.out.format("			return \"\" + //\n");
		compi.out.format("					\"compiled:\" + //\n");
		compi.out.format("					\"%s\" //\n", str);
		compi.out.format("			;\n");
		compi.out.format("		}\n");
	}

	private void compileGradientToStringWithLayoutAndValues(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public String toStringWithLayoutAndValues() {\n");
		compi.out.format("			return toStringBuilderWithLayoutAndValues(new StringBuilder(), 0).toString();\n");
		compi.out.format("		}\n");
	}

	private void compileGradientToStringBuilderWithLayout(Compilation compi) {
		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public StringBuilder toStringBuilderWithLayout(StringBuilder buf) {\n");
		compi.out.format("			return buf.append(toStringWithLayout());\n");
		compi.out.format("		}\n");
	}

	private void compileGradientToStringBuilderWithLayoutAndValues(Compilation compi) {
		Formatter aut = compi.out;

		aut.format("\n");
		aut.format("		@Override\n");
		aut.format("		public StringBuilder toStringBuilderWithLayoutAndValues("
				+ "StringBuilder buf, int indent) {\n");
		aut.format("			DecimalFormat format = createDecimalFormat();\n");
		aut.format("\n");
		aut.format("			String indentSpaces = SPACES.substring(0, indent);\n");
		aut.format("			buf.append(indentSpaces);\n");
		aut.format("			toStringBuilderWithLayout(buf).append(\"\\n\");\n");
		aut.format("\n");
		aut.format("			indent += 4;\n");
		aut.format("			indentSpaces = SPACES.substring(0, indent);\n");

		String storageVecFormat = "			toStringBuilderVec(" + //
				compi.storageVecFormat + ", buf, format).append(\"\\n\");\n";
		String paramMatFormat = "			toStringBuilderMat(" + //
				compi.paramMatFormat + ", buf, indent + 13, format).append(\"\\n\");\n";
		String paramVecFormat = "			toStringBuilderVec(" + //
				compi.paramVecFormat + ", buf, format).append(\"\\n\");\n";

		compi.resetVariableIndices();
		compi.storageVecIdx = -1;
		for (Operation op : compi.flattenedSrc) {
			Gradient grad = op.createGradient();

			aut.format("\n");
			aut.format("			buf.append(indentSpaces).append(\"" + grad.toStringWithLayout() + "\");\n");

			if (op instanceof Dense) {
				aut.format("			buf.append(\"\\n\");\n");

				aut.format("			buf.append(indentSpaces).append(\"    inp    : \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);

				aut.format("			buf.append(indentSpaces).append(\"    weights: \");\n");
				aut.format(paramMatFormat, ++compi.paramMatIdx);

				aut.format("			buf.append(indentSpaces).append(\"    bias   : \");\n");
				aut.format(paramVecFormat, ++compi.paramVecIdx);

				compi.methodDependencies.add("toStringBuilderMat");
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof Swish) {
				++compi.storageVecIdx;
				aut.format("			buf.append(\": inp: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof CombinedLossFunction) {
				aut.format("			buf.append(\": inp: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof ActivationFunction) {
				aut.format("			buf.append(\": inp: \");\n");
				aut.format(storageVecFormat, ++compi.storageVecIdx);
				compi.methodDependencies.add("toStringBuilderVec");
				compi.methodDependencies.add("toStringBuilderSca");

			} else if (op instanceof LossFunction) {
				// nothing to do for pure loss function

			} else {
				throwUnsupportedOperationException(op);
			}
		}

		aut.format("\n");
		aut.format("			return buf;\n");
		aut.format("		}\n");
	}

	private void compileGradientClear(final Compilation compi) {
		Consumer<int[]> storageVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String inpSize = "" + args[1];
				String varName = String.format(compi.storageVecFormat, idx);
				compileTemplate(compi, "\t", "assignVecSca", varName, "0.0f", inpSize);
			}
		};

		Consumer<int[]> paramMatCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String outSize = "" + args[1];
				String inpSize = "" + args[2];
				String varName = String.format(compi.paramMatFormat, idx);
				compileTemplate(compi, "\t", "assignMatSca", varName, "0.0f", outSize, inpSize);
			}
		};

		Consumer<int[]> paramVecCodeGenerator = new Consumer<int[]>() {
			public void accept(int[] args) {
				int idx = args[0];
				String inpSize = "" + args[1];
				String varName = String.format(compi.paramVecFormat, idx);
				compileTemplate(compi, "\t", "assignVecSca", varName, "0.0f", inpSize);
			}
		};

		Consumer<String> commentCodeGenerator = new Consumer<String>() {
			public void accept(String comment) {
				compi.out.format("			// %s\n", comment);
			}
		};

		compi.resetVariableIndices();

		compi.out.format("\n");
		compi.out.format("		@Override\n");
		compi.out.format("		public void clear() {\n");

		commentCodeGenerator.accept("input");
		storageVecCodeGenerator.accept(new int[] { compi.storageVecIdx, compi.src.getInputSize() });

		compileVariableHandling( //
				compi, //
				storageVecCodeGenerator, //
				paramMatCodeGenerator, //
				paramVecCodeGenerator, //
				commentCodeGenerator //
		);

		compi.out.format("		}\n");
	} // 1325

	private void compileGradientVariableDeclarations(Compilation compi) {
		compi.resetVariableIndices();

		String storageVecFormat = "		public final float[] " + compi.storageVecFormat + " = new float[%d];\n";
		String paramMatFormat = "		public final float[][] " + compi.paramMatFormat + " = new float[%d][%d];\n";
		String paramVecFormat = "		public final float[] " + compi.paramVecFormat + " = new float[%d];\n";
		String commentFormat = "		// %s\n";

		compi.out.format("\n");
		compi.out.format(commentFormat, "input");
		compi.out.format(storageVecFormat, compi.storageVecIdx, compi.src.getInputSize());

		compileVariableHandling( //
				compi, //
				storageVecFormat, //
				paramMatFormat, //
				paramVecFormat, //
				commentFormat//
		);
	}

	/**
	 * <code>replaceMany("A fox jumps over another fox.", "fox", "hedgehog", "jumps over", "looks at")</code>
	 * results in <code>"A hedgehog looks at another hedgehog."</code>.
	 */
	public static String replaceMany(String haystack, String... zippedReplaceMap) {
		if ((zippedReplaceMap.length & 1) == 1) {
			throw new IllegalArgumentException("" + //
					"zippedReplaceMap has " + zippedReplaceMap.length + " entries, but " + //
					"it must have an even number of entries, because every original " + //
					"must have its surrogate, e.g. (haystack, orig1, surr1, orig2, surr2)");
		}

		StringBuilder buf = new StringBuilder();
		boolean nothingReplaced = true;

		for (int pos = 0; pos < haystack.length(); pos++) {
			String foundOriginal = null;
			String foundSurrogate = null;

			for (int pairIdx = 0; pairIdx < zippedReplaceMap.length; pairIdx += 2) {
				String currOriginal = zippedReplaceMap[pairIdx + 0];
				String currSurrogate = zippedReplaceMap[pairIdx + 1];

				if (containsAt(haystack, currOriginal, pos)) {
					if (foundOriginal != null) {
						if (!currSurrogate.equals(foundSurrogate)) {
							System.out.println(haystack);

							throw new IllegalArgumentException(String.format( //
									"the originals \"%s\" and \"%s\" compete for replacement at position %d", //
									foundOriginal, currOriginal, pos));
						}
					} else {
						foundOriginal = currOriginal;
						foundSurrogate = currSurrogate;
					}
				}
			}

			if (foundOriginal == null) {
				buf.append(haystack.charAt(pos));
			} else {
				nothingReplaced = false;
				buf.append(foundSurrogate);
				pos += foundOriginal.length() - 1;
			}
		}

		if (nothingReplaced) {
			return haystack;
		} else {
			return buf.toString();
		}
	}

	/**
	 * Returns true if, and only if, the specified needle is contained in the
	 * haystack at the specified position.
	 */
	private static boolean containsAt(String haystack, String needle, int haystackPos) {
		if (haystack == null) {
			throw new NullPointerException("haystack may not be null");
		}

		if (needle == null) {
			throw new NullPointerException("needle may not be null");
		}

		if (haystackPos + needle.length() > haystack.length()) {
			return false;

		} else {
			for (int needlePos = 0; needlePos < needle.length(); haystackPos++, needlePos++) {
				if (haystack.charAt(haystackPos) != needle.charAt(needlePos)) {
					return false;
				}
			}

			return true;
		}
	}

	private static void throwUnsupportedOperationException(Operation op) {
		throw new UnsupportedOperationException("" + //
				op.getTypeShortname() + " " + //
				"(" + op.getClass().getName() + ")");
	}
}
