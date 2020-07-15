import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class TrainNetwork {


    public static void main(String[] args) throws IOException {
        var batchSize = 128;
        var emnistSet = EmnistDataSetIterator.Set.DIGITS;
        var emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
        var emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);


        int outputNum = EmnistDataSetIterator.numLabels(emnistSet);
        int rngSeed = 1;
        int numRows = 28;
        int numColumns = 28;


        var conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Adam())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();


        var network = new MultiLayerNetwork(conf);
        network.init();

//        var eachIterations = 1;
//        network.addListeners(new ScoreIterationListener(eachIterations));

        var epoches = 20;
        for (int i = 1; i <= epoches; i++){
            System.out.println("Epoch " + i + " / " + epoches);
            network.fit(emnistTrain);

            var eval = network.evaluate(emnistTest);
            System.out.println(eval.accuracy());
            System.out.println(eval.precision());
            System.out.println(eval.recall());
            System.out.println();
            File file = new File("model.m3_" + i);
            network.save(file, true);
        }

    }

}
