/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import java.io.File;
import java.io.IOException;
import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
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
        this.model = MultiLayerNetwork.load(new File(modelPath), true);
        this.afficherPredictions = afficherPredictions;
    }

    @Override
    public double evalSituation(SituationOthello s) {
        // Conversion du plateau en format CNN
        double[] features = s.getBoardAsArray();
        INDArray input = Nd4j.create(features).reshape(1, 1, 8, 8);
        
        // Prédiction
        INDArray output = model.output(input);
        double eval = output.getDouble(0);
        
        if(evaluePour == Joueur.BLANC) {
            eval = 1 - eval;
        }
        
        if (afficherPredictions) {
            System.out.printf("Prédiction CNN pour %s : %.2f%% de chances de gagner%n", 
                evaluePour, eval * 100);
        }

        return eval;
    }

    

    @Override
    public List<Joueur> joueursCompatibles() {
        // Liste des joueurs compatibles avec cet oracle
        return List.of(Joueur.NOIR, Joueur.BLANC);
    }

    @Override
    public Joueur getEvalueSituationApresCoupDe() {
        // Retourne le joueur pour lequel l'oracle évalue la situation
        return this.evaluePour;
    }

    @Override
    public void setEvalueSituationApresCoupDe(Joueur j) {
        // Définit le joueur pour lequel l'oracle évalue la situation
        this.evaluePour = j;
    }
    
}