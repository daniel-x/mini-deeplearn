package de.a0h.minideeplearn.gui.desktop;

import java.awt.BasicStroke;
import java.awt.Button;
import java.awt.Choice;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Image;
import java.awt.Label;
import java.awt.Panel;
import java.awt.RenderingHints;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import de.a0h.minideeplearn.datasetgenerator.DataSetGenerator;
import de.a0h.minideeplearn.datasetgenerator.vectordistribution.UniformVectorDistribution;
import de.a0h.minideeplearn.datasetgenerator.vectortransform.InsideCenteredSphere;
import de.a0h.minideeplearn.operation.optimizer.GradientDescentOptimizer;
import de.a0h.minideeplearn.operation.optimizer.Optimizer;
import de.a0h.minideeplearn.predefined.ActivationFunctionType;
import de.a0h.minideeplearn.predefined.Classifier;

@SuppressWarnings("serial")
public class MiniDeeplearnDesktopApp extends Frame implements WindowListener, ActionListener, ItemListener {

	private static final String TITLE = "Mini-Deeplearn";

	private static final String SIGN_PLAY = " ▶ ";

	private static final String SIGN_STOP = "▮▮";

	private static final String[] LEARNING_RATES = { //
			"0.00001", //
			"0.0001", //
			"0.001", //
			"0.003", //
			"0.01", //
			"0.03", //
			"0.1", //
			"0.3", //
			"1", //
			"3", //
			"10", //
	};

	private static final String[] HIDDEN_LAYER_ACT_FUNCS = { //
			ActivationFunctionType.IDENTITY.nameLowercase, //
			ActivationFunctionType.TANH.nameLowercase, //
			ActivationFunctionType.SIGMOID.nameLowercase, //
			ActivationFunctionType.SOFTPLUS.nameLowercase, //
			ActivationFunctionType.RELU.nameLowercase, //
			ActivationFunctionType.SWISH.nameLowercase, //
	};

	private static final String[] OUTPUT_LAYER_ACT_FUNCS = { //
			ActivationFunctionType.SIGMOID.nameLowercase, //
	};

	/**
	 * Lock for synchronizing gui and background activities.
	 */
	private final ReentrantLock guiLock = new ReentrantLock(true);

	private final Condition waitForRunnerExit = guiLock.newCondition();

	Button resetBtn = new Button(" ⚂ ");
	Button playStopBtn = new Button(SIGN_PLAY);
	Button stepBtn = new Button("▶︎▮");

	TextField epochTf = new TextField("0", 6);
	Choice learnRateCh = new Choice();
	Choice hiddenActCh = new Choice();
	Choice outputActCh = new Choice();

	// NetCanvas netCv;
	NetCanvas netCv;

	Random rnd = new Random(0);
	Classifier net;

	float[][] trainInp;
	float[][] trainTarget;
	float[][] testInp;
	float[][] testTarget;

	long epoch = 0;

	Thread runner;
	boolean shallRun = false;

	public MiniDeeplearnDesktopApp() {
		super(TITLE);

		setLayout(new GridBagLayout());

		GridBagConstraints gbc = new GridBagConstraints();

		resetBtn.setFont(getScaledFont(2.0f, 0));
		playStopBtn.setFont(getScaledFont(2.0f, 0));
		stepBtn.setFont(getScaledFont(2.0f, 0));

		Label epochLb = new Label("Epoch");
		epochTf.setFont(getScaledFont(1.5f, 0));
		epochTf.setEditable(false);

		Label learnRateLb = new Label("Learning Rate");
		learnRateCh.setFont(getScaledFont(1.5f, 0));
		for (String s : LEARNING_RATES) {
			learnRateCh.addItem(s);
		}
		learnRateCh.select("0.03");

		Label hiddenActLb = new Label("Hidden Layer Activation");
		hiddenActCh.setFont(getScaledFont(1.5f, 0));
		for (String s : HIDDEN_LAYER_ACT_FUNCS) {
			hiddenActCh.addItem(s);
		}
		hiddenActCh.select("swish");

		Label outputActLb = new Label("Output Layer Activation");
		outputActCh.setFont(getScaledFont(1.5f, 0));
		for (String s : OUTPUT_LAYER_ACT_FUNCS) {
			outputActCh.addItem(s);
		}
		outputActCh.select("sigmoid");

		Panel toolbarPl = new Panel();
		toolbarPl.setLayout(new FlowLayout());
		toolbarPl.add(resetBtn);
		toolbarPl.add(playStopBtn);
		toolbarPl.add(stepBtn);
		toolbarPl.add(new Label("     "));

		Panel selectionBar = new Panel();
		selectionBar.setLayout(new GridBagLayout());
		add(epochLb, selectionBar, gbc, 0, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(epochTf, selectionBar, gbc, 0, 1, 1, 2, 0, 0, GridBagConstraints.BOTH);
		add(new Label("     "), selectionBar, gbc, 1, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(learnRateLb, selectionBar, gbc, 2, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(learnRateCh, selectionBar, gbc, 2, 1, 1, 1, 0, 0, GridBagConstraints.NONE);
		add(new Label("     "), selectionBar, gbc, 3, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(hiddenActLb, selectionBar, gbc, 4, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(hiddenActCh, selectionBar, gbc, 4, 1, 1, 1, 0, 0, GridBagConstraints.NONE);
		add(new Label("     "), selectionBar, gbc, 5, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(outputActLb, selectionBar, gbc, 6, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);
		add(outputActCh, selectionBar, gbc, 6, 1, 1, 1, 0, 0, GridBagConstraints.NONE);
		add(new Label("     "), selectionBar, gbc, 7, 0, 1, 1, 0, 0, GridBagConstraints.BOTH);

		toolbarPl.add(selectionBar);

		add(toolbarPl, this, gbc, 0, 0, 2, 1, 100, 0, GridBagConstraints.BOTH);

		Label dataLb = new Label("DATA");
		dataLb.setFont(getScaledFont(1.2f, Font.BOLD));
		Panel dataPl = new Panel();
		dataPl.add(dataLb);
		add(dataPl, this, gbc, 0, 1, 1, 1, 0, 100, GridBagConstraints.BOTH);

		int[] shape = new int[] { 2, 4, 3, 1 };
		ActivationFunctionType hiddenLayerActFunc = ActivationFunctionType
				.valueOfLowercase(hiddenActCh.getSelectedItem());
		ActivationFunctionType outputLayerActFunc = ActivationFunctionType
				.valueOfLowercase(outputActCh.getSelectedItem());

		net = new Classifier(shape);
		net.setHiddenActivationFunction(hiddenLayerActFunc);
		net.setOutputActivationFunction(outputLayerActFunc);

		netCv = new NetCanvas(net);

		netCv.setFont(getScaledFont(1.2f, Font.PLAIN));
		add(netCv, this, gbc, 1, 1, 1, 1, 100, 100, GridBagConstraints.BOTH);

		initDataSets();
		netCv.netUpdated();

		pack();

		this.addWindowListener(this);
		resetBtn.addActionListener(this);
		playStopBtn.addActionListener(this);
		stepBtn.addActionListener(this);
		hiddenActCh.addItemListener(this);
		outputActCh.addItemListener(this);

		trainEpoch();
	}

	protected void initDataSets() {
		int combinedSampleCount = 600;
		int trainSize = combinedSampleCount * 2 / 3;
		int testSize = combinedSampleCount - trainSize;

		float bounds = 6.0f;
		float sphereDiam = (float) Math.sqrt(2.0f / Math.PI) * bounds;
		DataSetGenerator generator = new DataSetGenerator( //
				new UniformVectorDistribution(-bounds, +bounds), //
				new InsideCenteredSphere(sphereDiam) //
		);

		float[][][] trainSet = generator.generate(net.getInputSize(), net.getOutputSize(), trainSize, rnd);
		float[][][] testSet = generator.generate(net.getInputSize(), net.getOutputSize(), testSize, rnd);

		trainInp = trainSet[0];
		trainTarget = trainSet[1];
		testInp = testSet[0];
		testTarget = testSet[1];

		epoch = 0;

		netCv.trainInp = trainInp;
		netCv.trainTarget = trainTarget;
		netCv.testInp = testInp;
		netCv.testTarget = testTarget;
	}

	public static void main(String[] args) {
		new MiniDeeplearnDesktopApp().instanceMain(args);
	}

	private void instanceMain(String[] args) {
		setVisible(true);
		initIconImages();
	}

	protected void initIconImages() {
		int size = 16;
		List<Image> iconList = new ArrayList<>(4);
		for (int i = 0; i < 5; i++) {
			iconList.add(createIcon(size, size));
			size <<= 1;
		}

		setIconImages(iconList);
	}

	private Image createIcon(int w, int h) {
		Color bg = new Color(0, 0, 170);
		Color fg = new Color(128, 128, 255);
		Color fill = new Color(64, 64, 255);
		int strokeWidth = (w + h) * 6 / 100;

		Image img = createImage(w, h);
		Graphics2D g = (Graphics2D) img.getGraphics();
		RenderingHints rh = new RenderingHints( //
				RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setRenderingHints(rh);
		g.setStroke(new BasicStroke(strokeWidth));

		g.setColor(bg);
		g.fillRect(0, 0, w, h);

		g.setColor(fg);
		g.fillOval(w * 22 / 100, h * 27 / 100, w * 46 / 100, h * 46 / 100);

		g.drawLine(w * 15 / 100, h * 20 / 100, w * 45 / 100, h * 50 / 100);
		g.drawLine(w * 15 / 100, h * 80 / 100, w * 45 / 100, h * 50 / 100);
		g.drawLine(w * 45 / 100, h * 50 / 100, w * 83 / 100, h * 50 / 100);

		g.setColor(Color.GREEN);
		g.setColor(fill);
		strokeWidth = strokeWidth * 75 / 100;
		g.fillOval(w * 22 / 100 + strokeWidth, h * 27 / 100 + strokeWidth, w * 46 / 100 - 2 * strokeWidth,
				h * 46 / 100 - 2 * strokeWidth);

		g.dispose();

		return img;
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		Object src = e.getSource();

		guiLock.lock();

		try {
			if (src == hiddenActCh) {
				ActivationFunctionType actFuncType = ActivationFunctionType
						.valueOfLowercase(hiddenActCh.getSelectedItem());
				net.setHiddenActivationFunction(actFuncType);
				netCv.netUpdated();

			} else if (src == outputActCh) {
				ActivationFunctionType actFuncType = ActivationFunctionType
						.valueOfLowercase(outputActCh.getSelectedItem());
				net.setOutputActivationFunction(actFuncType);
				netCv.netUpdated();

			}
		} finally {
			guiLock.unlock();
		}
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		Object src = e.getSource();

		if (src == resetBtn) {
			resetButtonClicked();
		} else if (src == playStopBtn) {
			playStopBtnClicked();
		} else if (src == stepBtn) {
			stepBtnClicked();
		}
	}

	protected void stepBtnClicked() {
		guiLock.lock();

		try {
			trainEpochAndUpdateUi();
		} finally {
			guiLock.unlock();
		}
	}

	protected void trainEpochAndUpdateUi() {
		trainEpoch();
		uiUpdatesAfterTraining();
	}

	protected void playStopBtnClicked() {
		guiLock.lock();

		try {
			if (runner == null) {
				startBackgroundTraining();
			} else {
				stopBackgroundTraining();
			}
		} finally {
			guiLock.unlock();
		}
	}

	protected void startBackgroundTraining() {
		shallRun = true;
		runner = new Thread(new Runnable() {

			@Override
			public void run() {
				while (true) {
					guiLock.lock();
					try {

						if (!shallRun) {
							runner = null;
							waitForRunnerExit.signalAll();
							return;
						}

					} finally {
						guiLock.unlock();
					}

					try {
						trainEpochAndUpdateUi();
					} catch (Throwable t) {
						guiLock.lock();
						try {
							shallRun = false;
							runner = null;
							playStopBtn.setLabel(SIGN_PLAY);
							stepBtn.setEnabled(true);
							waitForRunnerExit.signalAll();
							throw t;
						} finally {
							guiLock.unlock();
						}
					}
				}
			}

		});

		playStopBtn.setLabel(SIGN_STOP);
		stepBtn.setEnabled(false);

		runner.start();
	}

	protected void stopBackgroundTraining() {
		shallRun = false;
		waitForRunnerExit.awaitUninterruptibly();

		playStopBtn.setLabel(SIGN_PLAY);
		stepBtn.setEnabled(true);
	}

	protected void resetButtonClicked() {
		guiLock.lock();
		try {
			// net.initWeightsAndClearBiases(rnd);
			net.initParams(rnd);
			netCv.netUpdated();
			epoch = 0;
			epochTf.setText("0");

		} finally {
			guiLock.unlock();
		}
	}

	private void uiUpdatesAfterTraining() {
		epochTf.setText(Long.toString(epoch));
		netCv.netUpdated();
	}

	private void trainEpoch() {
		String learningRateStr = learnRateCh.getSelectedItem();
		float learningRate = Float.parseFloat(learningRateStr);
		int batchSize = 20;

		Optimizer optimizer = new GradientDescentOptimizer();

		optimizer.run( //
				net, //
				trainInp, //
				trainTarget, //
				batchSize, //
				learningRate, //
				null //
		);

		epoch++;
	}

	@Override
	public void windowClosing(WindowEvent e) {
		System.exit(0);
	}

	@Override
	public void windowActivated(WindowEvent e) {
	}

	@Override
	public void windowClosed(WindowEvent e) {
	}

	@Override
	public void windowDeactivated(WindowEvent e) {
	}

	@Override
	public void windowDeiconified(WindowEvent e) {
	}

	@Override
	public void windowIconified(WindowEvent e) {
	}

	@Override
	public void windowOpened(WindowEvent e) {
	}

	protected static Font defaultFont = null;

	protected static Font getDefaultFont() {
		if (defaultFont == null) {
			Frame dummy = new Frame();
			dummy.setBounds(0, 0, 0, 0);
			dummy.setVisible(true);
			Graphics g = dummy.getGraphics();

			defaultFont = g.getFont();

			g.dispose();
			dummy.setVisible(false);
			dummy.dispose();
		}

		return defaultFont;
	}

	protected static Font getScaledFont(float factor, int additionalStyle) {
		Font defaultFont = getDefaultFont();

		int size = Math.round(defaultFont.getSize() * factor);
		int style = defaultFont.getStyle() | additionalStyle;
		Font font = new Font(defaultFont.getName(), style, size);

		return font;
	}

	public static void add(Component cmp, Container cnt, GridBagConstraints gbc, //
			int gridx, int gridy, int gridwidth, int gridheight, int weightx, int weighty, int fill) {
		gbc.gridx = gridx;
		gbc.gridy = gridy;
		gbc.gridwidth = gridwidth;
		gbc.gridheight = gridheight;
		gbc.weightx = weightx;
		gbc.weighty = weighty;
		gbc.fill = fill;

		cnt.add(cmp, gbc);
	}
}
