package de.a0h.minideeplearn.predefined;

import java.util.HashMap;
import java.util.NoSuchElementException;

public enum ActivationFunctionType {

	SIGMOID, //
	RELU, //
	TANH, //
	IDENTITY, //
	SOFTMAX, //
	SOFTPLUS, //
	SWISH, //
	;

	public final String nameLowercase;

	protected static HashMap<String, ActivationFunctionType> lowercaseMap = new HashMap<>();

	static {
		for (ActivationFunctionType func : values()) {
			lowercaseMap.put(func.nameLowercase, func);
		}
	}

	private ActivationFunctionType() {
		nameLowercase = name().toLowerCase();
	}

	public static ActivationFunctionType valueOfLowercase(String lowercaseName) {
		ActivationFunctionType result = lowercaseMap.get(lowercaseName);

		if (result == null) {
			throw new NoSuchElementException("no activation function with lower case name: " + lowercaseName);
		}

		return result;
	}
}
