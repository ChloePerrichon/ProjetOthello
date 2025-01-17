/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage.config_Othello;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @author francois
 */
public class ResumeResultat {

    private StatutSituation statutFinal;
    private List<CoupOthello> coupsJoues;

    public ResumeResultat(StatutSituation situationFinale, List<CoupOthello> coupsJoues) {
        this.statutFinal = situationFinale;
        this.coupsJoues = coupsJoues;
    }

    @Override
    public String toString() {
        return "ResumeResultat{" + "situationFinale=" + statutFinal
                + ", coupsJoues=" + Arrays.toString(coupsJoues.toArray()) + '}';
    }

    /**
     * @return the statutFinal
     */
    public StatutSituation getStatutFinal() {
        return statutFinal;
    }

    /**
     * @param statutFinal the statutFinal to set
     */
    public void setStatutFinal(StatutSituation statutFinal) {
        this.statutFinal = statutFinal;
    }

    /**
     * @return the coupsJoues
     */
    public List<CoupOthello> getCoupsJoues() {
        return coupsJoues;
    }

    /**
     * @param coupsJoues the coupsJoues to set
     */
    public void setCoupsJoues(List<CoupOthello> coupsJoues) {
        this.coupsJoues = coupsJoues;
    }

}
