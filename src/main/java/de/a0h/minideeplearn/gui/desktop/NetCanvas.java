package de.a0h.minideeplearn.gui.desktop;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import de.a0h.minideeplearn.predefined.Classifier;

@SuppressWarnings("serial")
public class NetCanvas extends Canvas {

	private static final int INSET_X = 10;

	private static final int OUTPUT_COL_W = 256 + 4;
	private static final int NEURON_SPACING_Y = 100;

	private static final int NEURON_ICON_SIZE = 32;
	private static final int NEURON_FULL_SIZE = 256;

	private final static int[] COLOR_SCHEME = new int[511];

	Color COLOR_BACKGROUND = new Color(205, 205, 205);
	Color COLOR_FOREGROUND = Color.BLACK;

	Classifier net;

	float[][] trainInp;
	float[][] trainTarget;
	float[][] testInp;
	float[][] testTarget;

	/**
	 * neuronImgs[layerIdx][neuronIdx];
	 */
	BufferedImage[][] neuronImgs;

	boolean discreteOutput = false;

	static {
		initColorScheme();
	}

	public NetCanvas(Classifier net) {
		this.net = net;

		allocNeuronImgs();

		setBackground(COLOR_BACKGROUND);
		setForeground(COLOR_FOREGROUND);
	}

	protected void allocNeuronImgs() {
		neuronImgs = new BufferedImage[net.getLayerCount()][];

		int lastLayerIdx = net.getLayerCount() - 1;
		for (int i = 0; i < lastLayerIdx; i++) {
			neuronImgs[i] = new BufferedImage[net.getLayerSize(i)];

			for (int j = 0; j < net.getLayerSize(i); j++) {
				neuronImgs[i][j] = new BufferedImage(NEURON_ICON_SIZE, NEURON_ICON_SIZE, BufferedImage.TYPE_INT_RGB);
			}
		}

		int i = lastLayerIdx;
		neuronImgs[i] = new BufferedImage[net.getLayerSize(i)];
		for (int j = 0; j < net.getLayerSize(i); j++) {
			neuronImgs[i][j] = new BufferedImage(NEURON_FULL_SIZE, NEURON_FULL_SIZE, BufferedImage.TYPE_INT_RGB);
		}
	}

	@Override
	public Dimension getPreferredSize() {
		return new Dimension(900, 500);
	}

	@Override
	public Dimension getMinimumSize() {
		return getPreferredSize();
	}

	public static int getColor(float v) {
		int idx;
		if (v <= -1) {
			idx = 0;
		} else if (v >= 1) {
			idx = COLOR_SCHEME.length - 1;
		} else {
			idx = Math.round((v + 1) / 2 * (COLOR_SCHEME.length - 1));
		}

		return COLOR_SCHEME[idx];
	}

	public void paintNeuronBorder(Graphics g, int x, int y, int size) {
		g.setColor(Color.BLACK);
		g.drawRect(x - 1, y - 1, size + 1, size + 1);
		g.drawRect(x - 2, y - 2, size + 3, size + 3);
	}

	public void update(Graphics g) {
		paint(g);
	}

	public void paint(Graphics g_) {
		Graphics2D g = (Graphics2D) g_;
		RenderingHints rh = new RenderingHints( //
				RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setRenderingHints(rh);

		int w = getWidth();
		int h = getHeight();

		int txtOffX = 10;
		int txtOffY = 20;

		int neuronSpacingX = getNeuronSpacingX();

		int inputColW = neuronSpacingX / 2;
		int outputColX = w - OUTPUT_COL_W;

		g.setColor(getBackground());
		g.fillRect(0, 0, w, h);

		g.setColor(new Color(180, 180, 180));
		g.fillRect(0, 0, inputColW, h);

		g.setColor(new Color(220, 220, 220));
		g.fillRect(outputColX, 0, OUTPUT_COL_W, h);

		g.setColor(getForeground());
		g.drawString("INPUT LAYER", txtOffX, txtOffY);
		g.drawString("HIDDEN LAYERS", inputColW + txtOffX, txtOffY);
		g.drawString("OUTPUT LAYER", outputColX + txtOffX, txtOffY);
		g.drawString("(single neuron)", outputColX + txtOffX, txtOffY + 20);

		paintNetwork(g);
	}

	protected int getNeuronSpacingX() {
		int w = getWidth();
		int outputColX = w - OUTPUT_COL_W;
		int layerCount = net.getLayerCount();

		return (outputColX - INSET_X) / (layerCount - 1);
	}

	private void paintNetwork(Graphics g) {
		paintNeuronImgs(g);
	}

	private void paintNeuronImgs(Graphics g) {
		int neuronSpacingX = getNeuronSpacingX();

		int neuronLocX = INSET_X;
		int neuronLocY;

		int outputLayerIdx = net.getLayerCount() - 1;

		for (int i = 0; i < outputLayerIdx; i++) {

			neuronLocY = NEURON_SPACING_Y;
			for (int j = 0; j < net.getLayerSize(i); j++) {
				paintNeuronIcon(g, neuronLocX, neuronLocY, i, j);
				neuronLocY += NEURON_SPACING_Y;
			}

			neuronLocX += neuronSpacingX;
		}

		int w = getWidth();
		neuronLocY = NEURON_SPACING_Y;
		neuronLocX = w - OUTPUT_COL_W + 2;
		for (int j = 0; j < net.getLayerSize(outputLayerIdx); j++) {
			paintNeuronFull(g, neuronLocX, neuronLocY, outputLayerIdx, j);
			neuronLocY += NEURON_SPACING_Y;
		}
	}

	private void paintNeuronFull(Graphics g, int x, int y, int layerIdx, int neuronIdx) {
		paintNeuronBorder(g, x, y, NEURON_FULL_SIZE);

		g.drawImage(neuronImgs[layerIdx][neuronIdx], x, y, NEURON_FULL_SIZE, NEURON_FULL_SIZE, null);
	}

	private void paintNeuronIcon(Graphics g, int x, int y, int layerIdx, int neuronIdx) {
		paintNeuronBorder(g, x, y, NEURON_ICON_SIZE);

		g.drawImage(neuronImgs[layerIdx][neuronIdx], x, y, null);
	}

	public void netUpdated() {
		updateNeuronImgs();
		paintNeuronImgs();
	}

	private void paintNeuronImgs() {
		Graphics g = getGraphics();

		if (g != null) {
			paintNeuronImgs(g);

			g.dispose();
		}
	}

	protected void updateNeuronImgs() {
		float intervalStart = -6;
		float intervalEnd = 6;
		float intervalDiam = intervalEnd - intervalStart;

		float[] input = new float[net.getLayerSize(0)];

		int lastLayerIdx = net.getLayerCount() - 1;

		for (int x = 0; x < NEURON_ICON_SIZE; x++) {
			for (int y = 0; y < NEURON_ICON_SIZE; y++) {
				int ɣ = NEURON_ICON_SIZE - 1 - y;
				input[0] = intervalStart + x * intervalDiam / (NEURON_ICON_SIZE - 1);
				input[1] = intervalStart + ɣ * intervalDiam / (NEURON_ICON_SIZE - 1);

				net.calcOutput(input);

				for (int i = 0; i < lastLayerIdx; i++) {
					float[] act = net.getActivity(i);
					for (int j = 0; j < net.getLayerSize(i); j++) {
						int rgb = getColor(act[j]);
						neuronImgs[i][j].setRGB(x, y, rgb);
					}
				}
			}
		}

		for (int x = 0; x < NEURON_FULL_SIZE; x++) {
			for (int y = 0; y < NEURON_FULL_SIZE; y++) {
				int ɣ = NEURON_FULL_SIZE - 1 - y;
				input[0] = intervalStart + x * intervalDiam / (NEURON_FULL_SIZE - 1);
				input[1] = intervalStart + ɣ * intervalDiam / (NEURON_FULL_SIZE - 1);

				net.calcOutput(input);

				int i = lastLayerIdx;
				float[] act = net.getActivity(i);

				for (int j = 0; j < net.getLayerSize(i); j++) {
					float v = act[j];

					v -= 0.5f; // because it's sigmoid

					if (discreteOutput) {
						v = v < 0 ? -1 : 1;
					}

					int rgb = getColor(v);
					neuronImgs[i][j].setRGB(x, y, rgb);
				}
			}
		}

		if (testInp != null) {
			int i = lastLayerIdx;
			Color negColor = new Color(getColor(-1));
			Color posColor = new Color(getColor(+1));
			for (int j = 0; j < net.getLayerSize(i); j++) {
				Graphics g_ = neuronImgs[i][j].getGraphics();
				Graphics2D g = (Graphics2D) g_;

				RenderingHints rh = new RenderingHints( //
						RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				g.setRenderingHints(rh);

				for (int k = 0; k < testInp.length; k++) {
					input = testInp[k];
					float[] y = testTarget[k];
					int realCategory = net.getRealityCategory(y);

					int px = Math.round((input[0] - intervalStart) * (NEURON_FULL_SIZE - 1) / intervalDiam);
					int py = Math.round((input[1] - intervalStart) * (NEURON_FULL_SIZE - 1) / intervalDiam);
					int pɣ = NEURON_FULL_SIZE - 1 - py;
					g.setColor(Color.WHITE);
					g.fillOval(px - 4, pɣ - 4, 9, 9);
					Color c = (realCategory == 0) ? posColor : negColor;
					g.setColor(c);
					g.fillOval(px - 3, pɣ - 3, 7, 7);
				}

				g.dispose();
			}
		}
	}

//	private void paintColorScheme(Graphics g) {
//		for (int i = 0; i < COLOR_SCHEME.length; i++) {
//			int rgb = COLOR_SCHEME[i];
//			Color c = new Color(rgb, false);
//			g.setColor(c);
//			g.drawLine(0, i, 30, i);
//		}
//	}

	protected static void initColorScheme() {
		Color neg = new Color(255, 192, 0);
		Color pos = new Color(95, 41, 203);

		int sizeH = COLOR_SCHEME.length / 2;
		float rStart = neg.getRed() / 255.0f;
		float gStart = neg.getGreen() / 255.0f;
		float bStart = neg.getBlue() / 255.0f;
		float rDelta = 1 - rStart;
		float gDelta = 1 - gStart;
		float bDelta = 1 - bStart;
		for (int i = 0; i < sizeH; i++) {
			float ratio = ((float) i) / sizeH;

			float r = rStart + ratio * rDelta;
			float g = gStart + ratio * gDelta;
			float b = bStart + ratio * bDelta;

			COLOR_SCHEME[i] = new Color(r, g, b).getRGB();
		}

		float rEnd = pos.getRed() / 255.0f;
		float gEnd = pos.getGreen() / 255.0f;
		float bEnd = pos.getBlue() / 255.0f;
		rDelta = 1 - rEnd;
		gDelta = 1 - gEnd;
		bDelta = 1 - bEnd;
		for (int i = 0; i < sizeH + 1; i++) {
			float ratio = ((float) i) / sizeH;

			float r = 1 - ratio * rDelta;
			float g = 1 - ratio * gDelta;
			float b = 1 - ratio * bDelta;

			COLOR_SCHEME[sizeH + i] = new Color(r, g, b).getRGB();
		}
	}
}
