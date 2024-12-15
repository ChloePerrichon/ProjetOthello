/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
/**
 *
 * @author cperrichon01
 */
package ProjetChloeTheo.Apprentissage;

import java.util.*;

public class OracleIntelligent implements Oracle {
    
    private Joueur evaluePour;
    
    public OracleIntelligent(Joueur evaluePour) {
        //super(List.of(Joueur.NOIR,Joueur.BLANC), evaluePour);
        this.evaluePour = evaluePour;
    }

    @Override
    public double evalSituation(SituationOthello s) {
        //todo
        
        
        return 0; // a modifier
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