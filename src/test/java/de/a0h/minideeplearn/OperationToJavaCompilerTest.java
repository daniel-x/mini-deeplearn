package de.a0h.minideeplearn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import de.a0h.minideeplearn.operation.MdlOperationConfig;
import de.a0h.minideeplearn.operation.MdlOperationConfig.StringConversionPrecision;
import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.compiler.JavaToClassCompiler;
import de.a0h.minideeplearn.operation.compiler.OperationToJavaCompiler;
import de.a0h.minideeplearn.operation.optimizer.GradientDescentOptimizer;
import de.a0h.minideeplearn.predefined.ActivationFunctionType;
import de.a0h.minideeplearn.predefined.Classifier;
import de.a0h.mininum.MnFuncs;

public class OperationToJavaCompilerTest {

	@Test
	public void testCompile() {
		// Classifier chainOriginal = new Classifier(3, 2, 1);
		Classifier chainOriginal = new Classifier(42, 64, 48, 32, 24, 16, 7);
		chainOriginal.setHiddenActivationFunction(ActivationFunctionType.RELU);

		OperationToJavaCompiler toJavaCompiler = new OperationToJavaCompiler();
		JavaToClassCompiler toClassCompiler = new JavaToClassCompiler();

		String source = toJavaCompiler.compile(chainOriginal, "CompiledChain", false).toString();
		saveToJavaFile(source);
		System.out.println(source);
		Class<? extends Operation> clazz = toClassCompiler.compileOperation(source);
		Operation chainCompiled = toClassCompiler.newInstance(clazz);

		MdlOperationConfig.stringConversionPrecision = StringConversionPrecision.EXACT;

		Random rnd = new Random(241);

		ensureOutputAndLossEqual(chainOriginal, chainCompiled);

		rnd.setSeed(0);
		chainOriginal.initParams(rnd);
		rnd.setSeed(0);
		chainCompiled.initParams(rnd);

		float[][] inp = new float[1][chainOriginal.getInputSize()];
		float[][] target = new float[1][chainOriginal.getOutputSize()];
		MnFuncs.assignGaussian(inp, rnd);
		target[0][0] = 1.0f;
//
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
//		ensureOutputAndLossEqual(chainOriginal, chainCompiled);
//
//		chainOriginal.calcOutput(inp[0]);
//		chainOriginal.calcLoss(inp[0], target[0]);
//		chainCompiled.calcOutput(inp[0]);
//		chainCompiled.calcLoss(inp[0], target[0]);
//
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
//		ensureOutputAndLossEqual(chainOriginal, chainCompiled);
//
//		train(chainOriginal, inp, target);
//		train(chainCompiled, inp, target);
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
//		ensureOutputAndLossEqual(chainOriginal, chainCompiled);
//
//		chainOriginal.calcOutput(inp[0]);
//		chainOriginal.calcLoss(inp[0], target[0]);
//		chainCompiled.calcOutput(inp[0]);
//		chainCompiled.calcLoss(inp[0], target[0]);
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
//		ensureOutputAndLossEqual(chainOriginal, chainCompiled);
//
//		train(chainOriginal, inp, target);
//		train(chainCompiled, inp, target);
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
//		ensureOutputAndLossEqual(chainOriginal, chainCompiled);
//
//		chainOriginal.calcOutput(inp[0]);
//		chainOriginal.calcLoss(inp[0], target[0]);
//		chainCompiled.calcOutput(inp[0]);
//		chainCompiled.calcLoss(inp[0], target[0]);
////		System.out.println();
////		System.out.println(chainOriginal);
////		System.out.println(chainCompiled);
// ensureOutputAndLossEqual(chainOriginal, chainCompiled);

		int repeatCount = 5000 * 1000;
		double dCompiled = getExecTimeCalcOutput(chainCompiled, inp, repeatCount);
		double dOriginal = getExecTimeCalcOutput(chainOriginal, inp, repeatCount);
		System.out.println(dOriginal);
		System.out.println(dCompiled);
		System.out.println(dOriginal / dCompiled);
	}

	protected static void train(Operation op, float[][] inp, float[][] target) {
		GradientDescentOptimizer opti = new GradientDescentOptimizer();

		float learningRate = 1f;

		for (int i = 0; i < 1; i++) {
			opti.run(op, inp, target, -1, learningRate, null);
		}
	}

	protected static double getExecTimeCalcOutput(Operation op, float[][] inp, int repeatCount) {
		long result = Long.MAX_VALUE;

		for (int i = 0; i < repeatCount; i++) {
			long t = System.nanoTime();

			op.calcOutput(inp[0]);

			t = System.nanoTime() - t;

			result = Math.min(result, t);
		}

		return ((double) result) / 1000;
	}

	private void ensureOutputAndLossEqual(Operation a, Operation b) {
		float[] outA = a.getOutput();
		float[] outB = b.getOutput();

		if (outA.length != outB.length) {
			Assert.fail(String.format("" + //
					"output differs in length: a.out.length=%d != b.out.length=%d", //
					outA.length, outB.length));
		}

		for (int i = 0; i < outA.length; i++) {
			if (outA[i] != outB[i]) {
				Assert.fail(String.format("" + //
						"output not equal in index %d, a.out[i]=%.8f != b.out[i]=%.8f", //
						i, outA[i], outB[i]));
			}
		}

		float lossA = a.getLoss();
		float lossB = b.getLoss();

		if (lossA != lossB) {
			Assert.fail(String.format("" + //
					"loss differs, a.loss=%.9f != b.loss=%.9f", //
					lossA, lossB));
		}
	}

	protected static void saveToJavaFile(String source) {
		String dstClassName = JavaToClassCompiler.extractFullyQualifiedClassName(source);
		String dstFilename = dstClassName.replace('.', '/') + ".java";

		// to avoid junit to pick up the IDE's compilation result, delete
		// build-generated class files
		String classFilename = "target/classes/" + dstFilename.replaceAll(".java$", ".class");
		new File(classFilename).delete();
		classFilename = "target/classes/" + dstFilename.replaceAll(".java$", "\\$CompiledGradient.class");
		new File(classFilename).delete();

		dstFilename = "src/main/java/" + dstFilename;

		File dstFile = new File(dstFilename);

		try {
			FileWriter fw = new FileWriter(dstFile, false);
			fw.write(source);
			fw.close();
		} catch (IOException e) {
			throw new RuntimeException("couldn't write java to file '" + dstFile.getAbsolutePath() + "'", e);
		}
	}
}
