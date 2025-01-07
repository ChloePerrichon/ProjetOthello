/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage.oracles;

/**
 *
 * @author chloe
 */

import ProjetChloeTheo.Apprentissage.config_Othello.Joueur;
import ProjetChloeTheo.Apprentissage.config_Othello.SituationOthello;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

public class OraclePerceptron implements Oracle{
    
    private Joueur evaluePour;
    private MultiLayerNetwork model;
    private boolean afficherPredictions; 
  
    
    public OraclePerceptron(Joueur evaluePour, String modelPath,boolean afficherPredictions) throws IOException {
        // Initialisation de l'oracle avec le joueur pour lequel il évalue
        this.evaluePour = evaluePour;
        // Chargement du modèle de réseau de neurones à partir du fichier
        this.model = MultiLayerNetwork.load(new File(modelPath), true);
        this.afficherPredictions = afficherPredictions;
    }
    
    
    

    @Override
    public double evalSituation(SituationOthello s) {
        // Extraction des caractéristiques de la situation de jeu
        double[] features = s.getBoardAsArray();
        // Conversion des caractéristiques en INDArray pour l'entrée du modèle
        INDArray input = Nd4j.create(features, new int[]{1, features.length});
        // Prédiction de la probabilité de victoire
        INDArray output = model.output(input);
        
        double eval =output.getDouble(0);
        if(evaluePour == Joueur.BLANC) {
            eval=1-eval;
        }
        
        // Afficher la prédiction si activé
        if (afficherPredictions){
            System.out.printf("Prédiction pour %s : %.2f%% de chances de gagner%n", evaluePour, eval * 100);
        }

        // Retourner la prédiction
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
