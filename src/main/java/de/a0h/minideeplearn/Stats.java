package de.a0h.minideeplearn;

import de.a0h.mininum.format.StringUtil;

public class Stats {

	enum ResultType {
		TRUE_POSITIVE, //
		TRUE_NEGATIVE, //
		FALSE_POSITIVE, //
		FALSE_NEGATIVE, //
	}

	/**
	 * This is the sum of the losses. Usually, you are interested in the average
	 * loss of the batch. To get this, use getLoss().
	 */
	public float lossSum;

	public int batchSize;

	public int[][] predictionResultCounts;

	public Stats(int categoryCount) {
		int resultTypeCount = ResultType.values().length;

		predictionResultCounts = new int[categoryCount][resultTypeCount];
	}

	public float getLoss() {
		return lossSum / batchSize;
	}

	public String toString() {
		return "Stats\n" + getStatsString();
	}

	private void append(StringBuilder buf, int v, int colWidth) {
		String str = Integer.toString(v);
		buf.append(StringUtil.leftPad(str, colWidth));
	}

	private void append(StringBuilder buf, float v, int colWidth) {
		String str = Float.toString(v);
		buf.append(StringUtil.leftPad(str, colWidth));
	}

	public void aggregate(int realCategory, int predictedCategory, float sampleLoss) {
		batchSize++;

		lossSum += sampleLoss;

		boolean correct = (predictedCategory == realCategory);

		// prediction is correct
		// cat y ŷ [ TP TN FP FN ]
		// . 0 . . [ .. 1. .. .. ]
		// . 1 . . [ .. 1. .. .. ]
		// . 2 . . [ .. 1. .. .. ]
		// . 3 1 1 [ 1. .. .. .. ]
		// . 4 . . [ .. 1. .. .. ]
		// . 5 . . [ .. 1. .. .. ]

		// prediction is incorrect
		// cat y ŷ [ TP TN FP FN ]
		// . 0 . . [ .. 1. .. .. ]
		// . 1 . . [ .. 1. .. .. ]
		// . 2 1 . [ .. .. .. 1. ]
		// . 3 . . [ .. 1. .. .. ]
		// . 4 . . [ .. 1. .. .. ]
		// . 5 . 1 [ .. .. 1. .. ]

		for (int i = 0; i < predictionResultCounts.length; i++) {
			ResultType resultType;

			if (correct) {
				if (i == predictedCategory) {
					resultType = ResultType.TRUE_POSITIVE;
				} else {
					resultType = ResultType.TRUE_NEGATIVE;
				}
			} else {
				if (i == predictedCategory) {
					resultType = ResultType.FALSE_POSITIVE;
				} else if (i == realCategory) {
					resultType = ResultType.FALSE_NEGATIVE;
				} else {
					resultType = ResultType.TRUE_NEGATIVE;
				}
			}

			predictionResultCounts[i][resultType.ordinal()]++;
		}
	}

	public float getAccuracy() {
		int correctResultCount = 0;

		for (int i = 0; i < predictionResultCounts.length; i++) {
			correctResultCount += predictionResultCounts[i][ResultType.TRUE_POSITIVE.ordinal()];
		}

		return ((float) correctResultCount) / batchSize;
	}

	public String getStatsString() {
		StringBuilder buf = new StringBuilder();

		buf.append("batchSize: ").append(batchSize).append("\n");
		buf.append("loss.....: ").append(getLoss()).append("\n");
		buf.append("accuracy.: ").append(getAccuracy()).append("\n");

		String s;
		String[] header = { //
				"cats↓\\results→", //
				"realPositive", //
				"realNegative", //
				"detectedPos", //
				"detectedNeg", //
				"truePos", //
				"trueNeg", //
				"falsePos", //
				"falseNeg", //
				"sensitivity", //
				"specificity", //
		};
		int colWidth = 0;
		for (String colName : header) {
			colWidth = Math.max(colWidth, colName.length());
		}
		colWidth++;
		for (String colName : header) {
			s = StringUtil.leftPad(colName, colWidth);
			buf.append(s);
		}
		buf.append("\n");

		for (int i = 0; i < predictionResultCounts.length; i++) {
			int[] currCategoryCounts = predictionResultCounts[i];

			int truePos = currCategoryCounts[ResultType.TRUE_POSITIVE.ordinal()];
			int falsePos = currCategoryCounts[ResultType.FALSE_POSITIVE.ordinal()];
			int trueNeg = currCategoryCounts[ResultType.TRUE_NEGATIVE.ordinal()];
			int falseNeg = currCategoryCounts[ResultType.FALSE_NEGATIVE.ordinal()];
			int realPos = truePos + falseNeg;
			int realNeg = falsePos + trueNeg;
			int detectedPos = truePos + falsePos;
			int detectedNeg = trueNeg + falseNeg;

			append(buf, i, colWidth);

			append(buf, realPos, colWidth);
			append(buf, realNeg, colWidth);
			append(buf, detectedPos, colWidth);
			append(buf, detectedNeg, colWidth);

			for (int j = 0; j < ResultType.values().length; j++) {
				append(buf, currCategoryCounts[j], colWidth);
			}

			append(buf, ((float) truePos) / realPos, colWidth);
			append(buf, ((float) trueNeg) / realNeg, colWidth);

			buf.append("\n");
		}

		return buf.toString();
	}
}
