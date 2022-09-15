package de.a0h.minideeplearn.datasetgenerator;

import java.util.Random;

import de.a0h.minideeplearn.datasetgenerator.vectordistribution.UniformVectorDistribution;
import de.a0h.minideeplearn.datasetgenerator.vectordistribution.VectorDistribution;
import de.a0h.minideeplearn.datasetgenerator.vectortransform.InsideCenteredSphere;
import de.a0h.minideeplearn.datasetgenerator.vectortransform.InsideFirstQuadrant;
import de.a0h.minideeplearn.datasetgenerator.vectortransform.VectorTransformation;

public class DataSetGenerator {

	VectorDistribution mInputDistribution;

	VectorTransformation mTransformation;

	public DataSetGenerator( //
			VectorDistribution inputDistribution, //
			VectorTransformation transformation //
	) {
		mInputDistribution = inputDistribution;
		mTransformation = transformation;
	}

	/**
	 * result[train=0,test=1][input=0,output=1][sampleIdx][vectorElementIdx]
	 */
	public static float[][][][] generateUniformToUnitSphere_TrainAndTestSets(int inputSize, int outputSize,
			int combinedSampleCount, Random rnd) {
		int trainSize = combinedSampleCount * 2 / 3;
		int testSize = combinedSampleCount - trainSize;

		DataSetGenerator generator = new DataSetGenerator( //
				new UniformVectorDistribution(-2.0f, +2.0f), //
				new InsideCenteredSphere(1.0f) //
		);

		float[][][] trainSet = generator.generate(inputSize, outputSize, trainSize, rnd);
		float[][][] testSet = generator.generate(inputSize, outputSize, testSize, rnd);

		return new float[][][][] { trainSet, testSet };
	}

	/**
	 * result[train=0,test=1][input=0,output=1][sampleIdx][vectorElementIdx]
	 */
	public static float[][][][] generateUniformToFirstQuadrant_TrainAndTestSets(int inputSize, int outputSize,
			int combinedSampleCount, Random rnd) {
		int trainSize = combinedSampleCount * 2 / 3;
		int testSize = combinedSampleCount - trainSize;

		DataSetGenerator generator = new DataSetGenerator( //
				new UniformVectorDistribution(-2.0f, +2.0f), //
				new InsideFirstQuadrant() //
		);

		float[][][] trainSet = generator.generate(inputSize, outputSize, trainSize, rnd);
		float[][][] testSet = generator.generate(inputSize, outputSize, testSize, rnd);

		return new float[][][][] { trainSet, testSet };
	}

	/**
	 * The returned result is a two-element array with result[0] being an array of
	 * input vectors and result[1] being the corresponding array of output vectors.
	 */
	public float[][][] generate(int inputSize, int outputSize, int sampleCount, Random rnd) {
		float[][] xAr = new float[sampleCount][inputSize];
		float[][] yAr = new float[sampleCount][outputSize];

		for (int i = 0; i < xAr.length; i++) {
			float[] x = xAr[i];
			float[] y = yAr[i];

			mInputDistribution.drawSample(x, rnd);
			mTransformation.transform(x, y, rnd);
		}

		return new float[][][] { xAr, yAr };
	}
}
