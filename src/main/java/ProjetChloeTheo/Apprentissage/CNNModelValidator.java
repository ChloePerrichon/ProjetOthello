/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import java.io.File;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author chloe
 */
public class CNNModelValidator {
    
    public static void validateModel(String modelPath) {
        try {
            // 1. Charger le modèle
            System.out.println("=== Validation du modèle CNN ===");
            MultiLayerNetwork model = MultiLayerNetwork.load(new File(modelPath), true);
            
            // 2. Test avec des situations connues
            testKnownSituations(model);
            
            // 3. Test de cohérence des prédictions
            testPredictionConsistency(model);
            
            // 4. Vérification des poids du réseau
            checkModelWeights(model);
            
            // 5. Test de performance sur un petit ensemble de validation
            testOnValidationSet(model);
            
        } catch (Exception e) {
            System.err.println("Erreur lors de la validation du modèle : " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testKnownSituations(MultiLayerNetwork model) {
        System.out.println("\n1. Test des situations connues:");
        
        // Création d'une situation initiale (plateau vide avec 4 pions au centre)
        double[] initialBoard = new double[64];
        // Position des pions initiaux (centre du plateau)
        initialBoard[27] = 1;  // Blanc
        initialBoard[28] = -1; // Noir
        initialBoard[35] = -1; // Noir
        initialBoard[36] = 1;  // Blanc
        
        INDArray input = Nd4j.create(initialBoard).reshape(1, 1, 8, 8);
        double prediction = model.output(input).getDouble(0);
        
        System.out.println("Prédiction situation initiale: " + prediction);
        System.out.println("Attendu: ~0.5 (situation équilibrée)");
        
        if (Math.abs(prediction - 0.5) > 0.2) {
            System.out.println("⚠️ ATTENTION: La prédiction pour la situation initiale semble anormale!");
        }
    }
    
    private static void testPredictionConsistency(MultiLayerNetwork model) {
        System.out.println("\n2. Test de cohérence des prédictions:");
        
        // Créer un plateau favorable aux noirs
        double[] favorableToBlack = new double[64];
        for (int i = 0; i < 64; i++) {
            favorableToBlack[i] = Math.random() < 0.7 ? -1 : 1; // Plus de pions noirs
        }
        
        INDArray inputBlack = Nd4j.create(favorableToBlack).reshape(1, 1, 8, 8);
        double predictionBlack = model.output(inputBlack).getDouble(0);
        
        // Inverser le plateau (favorable aux blancs)
        double[] favorableToWhite = new double[64];
        for (int i = 0; i < 64; i++) {
            favorableToWhite[i] = -favorableToBlack[i];
        }
        
        INDArray inputWhite = Nd4j.create(favorableToWhite).reshape(1, 1, 8, 8);
        double predictionWhite = model.output(inputWhite).getDouble(0);
        
        System.out.println("Prédiction situation favorable aux noirs: " + predictionBlack);
        System.out.println("Prédiction situation favorable aux blancs: " + predictionWhite);
        
        if (predictionBlack <= predictionWhite) {
            System.out.println("⚠️ ATTENTION: Les prédictions ne semblent pas cohérentes!");
        }
    }
    
    private static void checkModelWeights(MultiLayerNetwork model) {
        System.out.println("\n3. Analyse des poids du réseau:");
        
        // Vérifier que les poids ne sont pas tous proches de 0 ou tous similaires
        for (int i = 0; i < model.getnLayers(); i++) {
            INDArray weights = model.getLayer(i).getParam("W");
            if (weights != null) {
                double meanWeight = weights.meanNumber().doubleValue();
                double stdWeight = weights.stdNumber().doubleValue();
                
                System.out.printf("Couche %d - Moyenne des poids: %.5f, Écart-type: %.5f%n", 
                                i, meanWeight, stdWeight);
                
                if (stdWeight < 0.01) {
                    System.out.println("⚠️ ATTENTION: Faible variation des poids dans la couche " + i);
                }
            }
        }
    }
    
    private static void testOnValidationSet(MultiLayerNetwork model) {
        System.out.println("\n4. Test sur ensemble de validation:");
        
        try {
            // Charger ou créer un petit ensemble de validation
            String validationFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\validation.csv";
            DataSet validationSet = CNN.createDataset(validationFile);
            
            // Reshape pour CNN
            int numExamples = (int) validationSet.getFeatures().size(0);
            INDArray reshapedFeatures = validationSet.getFeatures().reshape(numExamples, 1, 8, 8);
            validationSet.setFeatures(reshapedFeatures);
            
            // Évaluation
            RegressionEvaluation eval = new RegressionEvaluation();
            INDArray predictions = model.output(validationSet.getFeatures());
            eval.eval(validationSet.getLabels(), predictions);
            
            System.out.println("MSE: " + eval.meanSquaredError(0));
            System.out.println("MAE: " + eval.meanAbsoluteError(0));
            System.out.println("Corrélation: " + eval.pearsonCorrelation(0));
            
            if (eval.pearsonCorrelation(0) < 0.3) {
                System.out.println("⚠️ ATTENTION: Faible corrélation sur l'ensemble de validation!");
            }
            
        } catch (Exception e) {
            System.out.println("Erreur lors du test sur l'ensemble de validation: " + e.getMessage());
        }
    }
}
