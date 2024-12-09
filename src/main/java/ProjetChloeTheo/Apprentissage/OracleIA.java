/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 *
 * @author chloé 
 */
public class OracleIA implements Oracle{

    private Joueur evaluePour;
    private MultiLayerNetwork model;
    
    public OracleIA(Joueur evaluePour,String modelPath) {
        this.evaluePour = evaluePour;
        try {
            this.model = VersionMeilleurIA.loadModel(modelPath);
        } catch (IOException ex) {
            throw new Error(ex);
        }
    }
    @Override
    public double evalSituation(SituationOthello s) {
        // Convertir la situation de jeu en entrée pour le modèle
        INDArray input = situationToINDArray(s);

        // Utiliser le modèle pour faire une prédiction
        INDArray output = model.output(input);

        // Retourner la prédiction
        return output.getDouble(0);
    }

    @Override
    public List<Joueur> joueursCompatibles() {
        return List.of(Joueur.NOIR,Joueur.BLANC);
    }

    @Override
    public Joueur getEvalueSituationApresCoupDe() {
        return this.evaluePour;
    }

    @Override
    public void setEvalueSituationApresCoupDe(Joueur j) {
        this.evaluePour = j;
    }
    
    // Méthode pour convertir une situation de jeu en INDArray
    private INDArray situationToINDArray(SituationOthello s) {
        // Assuming SituationOthello has a method to get the board as a 1D array of doubles
        double[] boardArray = s.getBoardAsArray();
        if (boardArray.length != 64) {
            throw new IllegalArgumentException("La taille du plateau doit être de 64 cases.");
        }

        // Convertir le tableau en INDArray
        INDArray input = Nd4j.create(boardArray, 1, 64);
        return input;
    }
    
}
