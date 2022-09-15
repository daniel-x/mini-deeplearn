package de.a0h.minideeplearn;

import java.util.Random;

import org.junit.Before;
import org.junit.Test;

import de.a0h.minideeplearn.operation.gradient.Gradient;
import de.a0h.minideeplearn.predefined.Classifier;
import de.a0h.mininum.MnFuncs;

public class ClassifierTest {

	Random rnd;

	@Before
	public void resetRnd() {
		rnd = new Random(0);
	}

	@Test
	public void testInitWeights() {
		testInitWeightsImpl();
	}

	public Classifier testInitWeightsImpl() {
		// int[] shape = new int[] { 2, 4, 3, 2 };
		int[] shape = new int[]{2, 4, 2, 2};
		Classifier net = new Classifier(shape);

		net.initParams(rnd);

		return net;
	}

	@Test
	public void testCalculateForward() {
		testCalculateForwardImpl();
	}

	public Classifier testCalculateForwardImpl() {
		Classifier net = testInitWeightsImpl();

		float[] x = new float[net.getInputSize()];

		MnFuncs.assignGaussian(x, rnd);
		net.calcOutput(x);

		return net;
	}

	// @Test
	// public void testCalculateGradient() {
	// testCalculateGradientImpl();
	// }
	//
	// public Classifier testCalculateGradientImpl() {
	// Classifier net = testCalculateForwardImpl();
	//
	// float[] y = new float[net.getOutputSize()];
	// y[0] = 1;
	//
	// Gradient grad = net.createGradient();
	// net.calcGradient(null, y, grad);
	//
	// return net;
	// }

	// @Test
	// public void testGradientChecking() {
	// rnd = new Random(1);
	//
	// Classifier net = new Classifier(new int[] { 1, 1 });
	// net.init(rnd);
	//
	// int sampleCount = 150;
	//
	// float bounds = 6.0f;
	// float sphereDiam = (float) Math.sqrt(2.0f / Math.PI) * bounds;
	// DataSetGenerator generator = new DataSetGenerator( //
	// new UniformVectorDistribution(-bounds, +bounds), //
	// new InsideCenteredSphere(sphereDiam) //
	// );
	//
	// float[][][] dataSet = generator.generate(net.getInputSize(),
	// net.getOutputSize(), sampleCount, rnd);
	//
	// float[][] xAr = dataSet[0];
	// float[][] yAr = dataSet[1];
	//
	// Gradient gradAnalytic = net.createGradient();
	// Gradient gradNumeric = net.createGradient();
	// Gradient gradErr = net.createGradient();
	//
	// float epsilon = 0.0000001f;
	//
	// for (int i = 0; i < xAr.length; i++) {
	// float[] x = xAr[i];
	// float[] y = yAr[i];
	//
	// net.calcOutput(x);
	// net.calcGradient(null, y, gradAnalytic);
	//
	// net.calcGradientNumerically(null, y, gradNumeric, epsilon);
	//
	// gradAnalytic.assignTo(gradErr);
	// gradErr.sub(gradNumeric);
	//
	// float err = gradErr.weightsMaxAbs();
	// if (err > 0.0001f) {
	// System.out.println("#########################################");
	// System.out.println("i: " + i);
	// System.out.println("err: " + err);
	// System.out.println("");
	// System.out.println("gradAnalytic: " + gradAnalytic.toStringWithAll());
	// System.out.println("gradNumeric.: " + gradNumeric.toStringWithAll());
	// System.out.println("gradErr.....: " + gradErr.toStringWithAll());
	// break;
	// }
	// }
	// }

	// @Test
	// public void testTrainEpoch() {
	// testTrainEpochImpl();
	// }
	//
	// public Classifier testTrainEpochImpl() {
	// Classifier net = testInitWeightsImpl();
	//
	// int sampleCount = 1500;
	// float[][][][] dataSets;
	// dataSets = DataSetGenerator.generateUniformToUnitSphere_TrainAndTestSets(
	// //
	//// dataSets =
	// ExampleDataSetGenerator.generateUniformToFirstQuadrant_TrainAndTestSets(
	// //
	// net.getInputSize(), //
	// net.getOutputSize(), //
	// sampleCount, //
	// rnd //
	// );
	// float[][][] trainSet = dataSets[0];
	// float[][][] testSet = dataSets[1];
	// float[][] trainInp = trainSet[0];
	// float[][] trainTarget = trainSet[1];
	// float[][] testInp = testSet[0];
	// float[][] testTarget = testSet[1];
	//
	// Stats testStats;
	// testStats = net.calcStats(testInp, testTarget);
	// System.out.println("test stats before trainig:\n" +
	// testStats.getStatsString());
	//
	// float[][] testPred = new float[testTarget.length][testTarget[0].length];
	// net.calcOutput(testInp, testPred);
	//
	// ClosingAwaitableFrame f = new
	// ClosingAwaitableFrame(getClass().getSimpleName());
	// MlResultDisplayCanvas mlCanvas = new MlResultDisplayCanvas();
	// mlCanvas.setData(testInp, testTarget, testPred);
	// f.add(mlCanvas);
	//
	// f.pack();
	// f.setVisible(true);
	//
	// float learningRateMin = 0.0001f;
	// float learningRateAdaptive = 0.1f;
	// // float learningRateAdaptionFactor = 0.999f;
	// float learningRateAdaptionFactor = 1;
	// float adaption = 1;
	//
	// for (int epoch = 0; epoch < 100000; epoch++) {
	// adaption *= learningRateAdaptionFactor;
	// float learningRate = learningRateMin + learningRateAdaptive * adaption;
	//
	// if (epoch % 1000 == 0) {
	// System.out.println("epoch: " + epoch + " learningRate: " + learningRate);
	// for (int i = 0; i < net.wei.length; i++) {
	// System.out.println(Format.toString(net.wei[i]));
	// }
	// }
	//
	// net.calcOutput(testInp, testPred);
	// mlCanvas.paint();
	// // f.sleepUniterruptibly(10);
	//
	// if (f.closing) {
	// break;
	// }
	//
	// batchGrad.clear();
	// net.trainEach(trainInp, trainTarget, learningRate, batchGrad,
	// sampleGrad);
	// }
	//
	// testStats = net.calcStats(testInp, testTarget);
	// System.out.println("test stats after trainig:\n" +
	// testStats.getStatsString());
	//
	// testStats = net.calcStats(trainInp, trainTarget);
	// System.out.println("train stats after trainig:\n" +
	// testStats.getStatsString());
	//
	// // f.awaitWindowClosingUninterruptibly();
	//
	// return net;
	// }
}
