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

/**
 *
 * @author toliveiragaspa01
 */
public class OracleIA implements Oracle{

    private Joueur evaluePour;
    private MultiLayerNetwork vraiOracle;
    
    public OracleIA(Joueur evaluePour,File sauvegarde) {
        try {
            this.evaluePour = evaluePour;
            this.vraiOracle = MultiLayerNetwork.load(sauvegarde, true);
        } catch (IOException ex) {
            throw new Error(ex);
        }
    }
     @Override
    public double evalSituation(SituationOthello s) {
        // transformer la situation s en IndArray
         INDArray situation = null;
//         return VersionMeilleurIA.makePrediction(this.vraiOracle,situation);
        return 0.5;
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
    
}
