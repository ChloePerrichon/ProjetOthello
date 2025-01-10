package ProjetChloeTheo.Apprentissage.modeles;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author chloe
 */
public class Perceptron {
    private static MultiLayerNetwork model;
    
    
    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba8000OMCNN-OPPER.csv";
            
            // Paramètres du modèle
            int seed = 123;
            double learningRate = 0.001;
            int numEpochs = 30;
            int batchSize = 128;
            
            // Création et préparation du dataset
            DataSet dataset = createDataset(csvFilePath);
            System.out.println("Dataset created with " + dataset.numExamples() + " examples.");

            // Division du dataset en ensembles d'entrainement et de test
            dataset.shuffle(123);
            int trainSize = (int) (dataset.numExamples() * 0.8);
            SplitTestAndTrain splits = dataset.splitTestAndTrain(trainSize);
            DataSet trainData = splits.getTrain();
            DataSet testData = splits.getTest();

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), 128);
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), 128);

            

            // Création et entraînement du modèle
            //model = createModel(seed, learningRate);
            
            // Entraînement du modèle
            System.out.println("Starting training...");
            //trainModel(model, trainIterator, numEpochs);
           
            // Sauvegarde du modèle
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron2-model.zip";
            //saveModel(model, modelPath);
            
            // Évaluation du modèle
            System.out.println("\nÉvaluation du modèle...");
            evaluateModel(modelPath, testIterator);
                    

        } catch (IOException | IllegalArgumentException e) {
            e.printStackTrace();
        }
    }
    
    //Méthode pour entrainer le modèle
    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            trainIterator.reset();
            int batchNum = 0;
            while (trainIterator.hasNext()) {
                DataSet batch = trainIterator.next();
                model.fit(batch);
                if (batchNum % 10 == 0) {
                    System.out.printf("Epoch %d, Batch %d: Score = %.4f%n", 
                        epoch + 1, batchNum, model.score());
                }
                batchNum++;
            }
            System.out.println("Completed epoch " + (epoch + 1));
        }
    }
    // Méthode améliorée d'évaluation
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
         MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath));
         System.out.println("Modèle chargé depuis: " + modelPath);

         RegressionEvaluation eval = new RegressionEvaluation();
         int totalPredictions = 0;
         int correctPredictions = 0;
         double mseSum = 0.0;
         double threshold = 0.5;

         testIterator.reset();
         while (testIterator.hasNext()) {
             DataSet batch = testIterator.next();
             INDArray features = batch.getFeatures();
             INDArray labels = batch.getLabels();

             INDArray predictions = model.output(features);
             eval.eval(labels, predictions);

             for (int i = 0; i < predictions.length(); i++) {
                 totalPredictions++;
                 double predicted = predictions.getDouble(i);
                 double actual = labels.getDouble(i);

                 if ((predicted >= threshold && actual >= threshold) ||
                     (predicted < threshold && actual < threshold)) {
                     correctPredictions++;
                 }

                 mseSum += Math.pow(predicted - actual, 2);
             }
         }

         double accuracy = (double) correctPredictions / totalPredictions * 100;
         double mse = mseSum / totalPredictions;
         double rmse = Math.sqrt(mse);

         System.out.println("\nRésultats de l'évaluation du modèle Perceptron :");
         System.out.println("----------------------------------------");
         System.out.printf("Précision de classification : %.2f%% (%d/%d)%n", 
                 accuracy, correctPredictions, totalPredictions);
         System.out.printf("Erreur quadratique moyenne (MSE) : %.4f%n", mse);
         System.out.printf("RMSE : %.4f%n", rmse);
         System.out.printf("Coefficient de corrélation : %.4f%n", eval.pearsonCorrelation(0));
         System.out.printf("MAE : %.4f%n", eval.averageMeanAbsoluteError());
     }



     

    // Fonction de création du dataset à partir d'un fichier CSV
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                String[] values = line.split(",");
                if (values.length != 65) { // Ajusté pour 65 colonnes (64 entrées + 1 sortie)
                    System.err.println("Line " + lineNumber + " has " + values.length + " columns: " + line);
                    continue; // Sauter cette ligne
                }

                // Créer l'entrée INDArray avec 64 valeurs
                INDArray input = Nd4j.zeros(1, 64);
                boolean validInput = true;
                for (int i = 0; i < 64; i++) {
                    try {
                        input.putScalar(i, Double.parseDouble(values[i]));
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid number at line " + lineNumber + " column " + i + ": " + values[i]);
                        validInput = false;
                        break;
                    }
                }
                if (!validInput) continue;

                // Créer la sortie INDArray avec 1 valeur
                INDArray output = Nd4j.zeros(1, 1);
                try {
                    output.putScalar(0, Double.parseDouble(values[64]));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number at line " + lineNumber + " column 64: " + values[64]);
                    continue;
                }
                
                // Ajouter les entrées et sorties validées à leurs listes respectives
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        if (inputList.isEmpty() || outputList.isEmpty()) {
            throw new IllegalArgumentException("No valid data found in the CSV file.");
        }

        // Empiler les entrées et sorties en un seul INDArray
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);
        
        // Vérification de la forme des tableaux
        if (input.rank() != 2 || output.rank() != 2) {
            throw new IllegalStateException("Input or output has invalid shape. Rank: " + input.rank() + ", " + output.rank());
        }

        DataSet dataset = new DataSet(input, output);

        // Normalisation du dataset
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);

        return dataset;
    }
    
    // Fonction de création du modèle de réseau neuronal
    public static MultiLayerNetwork createModel(int seed, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed) // reproduit les résultats
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//optimisation
            .updater(new Nesterovs(learningRate, 0.9)) // mise à jours des poids
            .weightInit(WeightInit.XAVIER) //initialisation des poids 
            .list()
            //Première couche cachée
            .layer(new DenseLayer.Builder()
                .nIn(64) // Nombre d'entrées (64 cases du plateau Othello)
                .nOut(256) //256 neuronnes
                .activation(Activation.RELU)
                .build())
            //Deuxième couche cachée
            .layer(new DenseLayer.Builder()
                .nOut(256) //256 neuronnes 
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Utilisation de la MSE pour la régression
                .nOut(1) // Une seule sortie pour la probabilité de victoire
                .activation(Activation.IDENTITY) // Activation identité pour la régression
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();// initialisation des paramètres du modèle
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }
    
    
    
    
   
    
    // Fonction d'enregistrement du modèle
    public static void saveModel(MultiLayerNetwork model, String modelPath) throws IOException {
        System.out.println("Saving the model...");
        model.save(new File(modelPath));
        System.out.println("Model saved to " + modelPath);
    }

    
   
    
   

    
}

