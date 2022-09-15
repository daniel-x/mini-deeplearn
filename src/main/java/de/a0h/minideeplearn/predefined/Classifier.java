package de.a0h.minideeplearn.predefined;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.minideeplearn.operation.activation.Identity;
import de.a0h.minideeplearn.operation.activation.Relu;
import de.a0h.minideeplearn.operation.activation.Sigmoid;
import de.a0h.minideeplearn.operation.activation.Softmax;
import de.a0h.minideeplearn.operation.activation.Softplus;
import de.a0h.minideeplearn.operation.activation.Swish;
import de.a0h.minideeplearn.operation.activation.Tanh;
import de.a0h.minideeplearn.operation.composite.Chain;
import de.a0h.minideeplearn.operation.layer.Dense;
import de.a0h.minideeplearn.operation.loss.SigmoidWithCrossEntropyLoss;
import de.a0h.minideeplearn.operation.loss.SoftmaxWithCrossEntropyLoss;
import de.a0h.mininum.MnFuncs;

public class Classifier extends Chain {

	public Classifier(int... shape) {
		if (shape.length == 0) {
			throw new IllegalArgumentException(
					"you must provide a non-empty shape of type int[] for the neural layer sizes");
		}

		for (int i = 0; i < shape.length - 1; i++) {
			int inpSize = shape[i];
			int outSize = shape[i + 1];

			add(new Dense(inpSize, outSize));

			if (i < shape.length - 2) {
				add(new Tanh(outSize));

			} else {
				if (outSize == 1) {
					add(new SigmoidWithCrossEntropyLoss(outSize));
				} else {
					add(new SoftmaxWithCrossEntropyLoss(outSize));
				}
			}
		}
	}

	public void setHiddenActivationFunction(ActivationFunctionType actFuncType) {
		for (int i = 1; i < size() - 2; i += 2) {
			Operation el = get(i);

			if (actFuncType == ActivationFunctionType.SIGMOID) {
				el = new Sigmoid(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.TANH) {
				el = new Tanh(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.RELU) {
				el = new Relu(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.SOFTPLUS) {
				el = new Softplus(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.SOFTMAX) {
				el = new Softmax(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.IDENTITY) {
				el = new Identity(el.getInputSize());
			} else if (actFuncType == ActivationFunctionType.SWISH) {
				el = new Swish(el.getInputSize());
			} else {
				throw new IllegalArgumentException(
						"activation function type not supported for hidden layers: " + actFuncType);
			}

			set(i, el);
		}
	}

	public void setOutputActivationFunction(ActivationFunctionType actFuncType) {
		int i = size() - 1;

		Operation el = get(i);
		int outSize = el.getOutputSize();

		if (actFuncType == ActivationFunctionType.SIGMOID) {
			el = new SigmoidWithCrossEntropyLoss(outSize);

		} else if (actFuncType == ActivationFunctionType.SOFTMAX) {
			if (outSize == 1) {
				throw new IllegalArgumentException("" + //
						"softmax output activation doesn't make sense for output size 1." + //
						" use sigmoid instead.");
			}

			el = new SigmoidWithCrossEntropyLoss(outSize);

		} else {
			throw new IllegalArgumentException(
					"activation function type not supported for output layer: " + actFuncType);
		}

		set(i, el);
	}

	public static int getRealityCategory(float[] target) {
		int result;

		if (target.length == 1) {
			result = (target[0] == 1.0f) ? 0 : 1;
		} else {
			result = MnFuncs.onehotToIndex(target);
		}

		return result;
	}

	public static int getPredictedCategory(float[] output) {
		int result;

		if (output.length == 1) {
			result = (output[0] >= 0.5f) ? 0 : 1;
		} else {
			result = MnFuncs.idxMax(output);
		}

		return result;
	}

	public float[] calcOutput(float[] inp) {
		return super.calcOutput(inp);
	}

	public int getLayerCount() {
		return 1 + size() / 2;
	}

	public int getLayerSize(int layerIdx) {
		if (layerIdx == 0) {
			return get(0).getInputSize();
		} else {
			layerIdx = layerIdx * 2 - 1;
			return get(layerIdx).getOutputSize();
		}
	}

	public float[] getActivity(int layerIdx) {
		if (layerIdx == 0) {
			throw new IllegalArgumentException("layer 0 is the input activity and it's not supported by this method");
		} else {
			layerIdx = layerIdx * 2 - 1;
			return get(layerIdx).getOutput();
		}
	}

	public static int[] createShape(int inputSize, int outputSize, int layerCount) {
		int[] result = new int[layerCount];

		int iMax = layerCount - 1;
		for (int i = 0; i <= iMax; i++) {
			int size = (2 * inputSize * (iMax - i) / iMax) + (2 * outputSize * i / iMax);
			size = (size + 1) / 2;

			result[i] = size;
		}

		return result;
	}
}