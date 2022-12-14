package de.a0h.minideeplearn.operation.compiler;

import de.a0h.minideeplearn.operation.Operation;
import de.a0h.ontheflyjavacompiler.OnTheFlyJavaCompiler;

/**
 * Suitable for compiling simple java source code classes (e.g. that that was
 * generated by the {@link OperationToJavaCompiler}) to bytecode and then load
 * it into the JVM as a Class object.
 */
public class JavaToClassCompiler extends OnTheFlyJavaCompiler {

	public Class<? extends Operation> compileOperation(String source) {
		String className = extractFullyQualifiedClassName(source);

		return compileOperation(className, source, true);
	}

	public Class<? extends Operation> compileOperation(String source, boolean ignoreWarnings) {
		String className = extractFullyQualifiedClassName(source);

		return compileOperation(className, source, true);
	}

	public Class<? extends Operation> compileOperation(String className, String source) {
		return compileOperation(className, source, true);
	}

	public Class<? extends Operation> compileOperation(String className, String source, boolean ignoreWarnings) {
		Class<?> clazz = compile(className, source, ignoreWarnings);

		@SuppressWarnings("unchecked")
		Class<? extends Operation> opClazz = (Class<? extends Operation>) clazz;

		return opClazz;
	}
}
