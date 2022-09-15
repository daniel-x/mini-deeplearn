package de.a0h.minideeplearn.operation;

import de.a0h.minideeplearn.operation.MdlOperationConfig.StringConversionPrecision;
import de.a0h.mininum.format.DecimalFormatWithPadding;
import de.a0h.mininum.format.NumberStats;

public class OperationUtil {

	public static void throwLossNotProvidedException(Class<?> clazz) {
		throwLossNotProvidedException(clazz.getName());
	}

	public static void throwLossNotProvidedException(String className) {
		throw new UnsupportedOperationException("" + //
				"loss not provided: don't call this method on operations " + //
				"of type " + className + " or their compiled versions, " + //
				"because they don't provide a loss");
	}

	public static void throwOutputNotProvidedException(Class<?> clazz) {
		throwOutputNotProvidedException(clazz.getName());
	}

	public static void throwOutputNotProvidedException(String className) {
		throw new UnsupportedOperationException("" + //
				"output not provided: don't call this method on operations " + //
				"of type " + className + " or their compiled versions, " + //
				"because they don't provide an output");
	}

	public static DecimalFormatWithPadding getFormat(NumberStats numberStats) {
		DecimalFormatWithPadding format;

		if (MdlOperationConfig.stringConversionPrecision == StringConversionPrecision.EXACT) {
			format = numberStats.getFormatExact();
		} else if (MdlOperationConfig.stringConversionPrecision == StringConversionPrecision.PRETTY) {
			format = numberStats.getFormatPretty();
		} else {
			throw new UnsupportedOperationException("" + //
					"unknown " + StringConversionPrecision.class.getSimpleName() + " in " + //
					MdlOperationConfig.class.getSimpleName() + ": " + //
					MdlOperationConfig.stringConversionPrecision);
		}

		return format;
	}

}
