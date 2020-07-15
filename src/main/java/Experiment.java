import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;

public class Experiment {

    public static void main(String[] args) throws IOException {
        var network = MultiLayerNetwork.load(getFile("model.m2_11"), true);
        var image = loadImage("image.bmp");

        INDArray output = network.output(image);
        double[] results = output.toDoubleVector();
        for (int i = 0; i < results.length; i++) {
            System.out.println(String.format("Number %d -> %s%%", i, (int) (results[i] * 100)));
        }
    }

    private static INDArray loadImage(String imageName) throws IOException {
        NativeImageLoader nativeImageLoader = new NativeImageLoader(28, 28, 1);
        INDArray indArray = nativeImageLoader.asRowVector(getFile(imageName));
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);
        dataNormalization.transform(indArray);
        return indArray.reshape(1, 784);
    }

    private static File getFile(String fileName) {
        return new File(fileName);
    }

}
