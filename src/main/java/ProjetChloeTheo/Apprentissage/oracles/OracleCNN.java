/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage.oracles;

import ProjetChloeTheo.Apprentissage.config_Othello.Joueur;
import ProjetChloeTheo.Apprentissage.config_Othello.SituationOthello;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author chloe
 */
public class OracleCNN implements Oracle {
    private Joueur evaluePour;
    private MultiLayerNetwork model;
    private boolean afficherPredictions;
   
    
    public OracleCNN(Joueur evaluePour, String modelPath, boolean afficherPredictions) throws IOException {
        this.evaluePour = evaluePour;
        this.model = MultiLayerNetwork.load(new File(modelPath), true); // chargement du modèle CNN
        this.afficherPredictions = afficherPredictions;
    }
    
    @Override
    public double evalSituation(SituationOthello s) {
        // Conversion du plateau en format CNN donc matrice 8x8
        double[] features = s.getBoardAsArray();
        INDArray input = Nd4j.zeros(1, 1, 8, 8);
        for (int i = 0; i < 64; i++) {
            int row = i / 8;
            int col = i % 8;
            input.putScalar(new int[]{0, 0, row, col}, features[i]);
        }
        
        // Prédiction
        INDArray output = model.output(input);
        double eval = output.getDouble(0);
        
        // Ajustement pour le joueur blanc
        if (evaluePour == Joueur.BLANC) {
        eval = 1 - eval;
        
        }
        
        // Affichage de la prédiction
        if (afficherPredictions) {
            System.out.printf("Prédiction CNN pour %s : %.2f%% de chances de gagner%n", 
                evaluePour, eval * 100);
        }
        
        return eval; 
    }
    
    @Override
    public List<Joueur> joueursCompatibles() {
        return List.of(Joueur.NOIR, Joueur.BLANC);
    }
    
    @Override
    public Joueur getEvalueSituationApresCoupDe() {
        return this.evaluePour;
    }
    
    @Override
    public void setEvalueSituationApresCoupDe(Joueur j) {
        this.evaluePour = j;
    }
}