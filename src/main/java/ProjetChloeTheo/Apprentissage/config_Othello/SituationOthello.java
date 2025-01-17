/*
Copyright 2000- Francois de Bertrand de Beuvron

This file is part of CoursBeuvron.

CoursBeuvron is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CoursBeuvron is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CoursBeuvron.  If not, see <http://www.gnu.org/licenses/>.
 */
package ProjetChloeTheo.Apprentissage.config_Othello;

import ProjetChloeTheo.configuration_jeu_othello.Case;
import ProjetChloeTheo.configuration_jeu_othello.Damier;

/**
 * Un "proxy" de la classe Damier1 existante qui implementait le jeu d'othello.
 * @author francois
 */
public class SituationOthello {
    
    private Damier damierReel;

    public SituationOthello(Damier damierReel) {
        this.damierReel = damierReel;
    }

    public Damier getDamierReel() {
        return damierReel;
    }
    
    public String toCSV() {
        StringBuilder res = new StringBuilder();
        for (int lig = 0 ; lig < 8 ; lig ++) {
            for (int col = 0 ; col < 8 ; col ++) {
                Case cur = this.damierReel.getVal(lig, col);
                int val;
                if(cur == Case.VIDE) { //si la case est vide c'est 0
                    val = 0;
                } else if (cur == Case.NOIR) { //si la case est noir c'est 1
                    val = 1;
                } else {
                    val = -1; //si la case est blanche c'est -1
                }
                res.append(val);
                if (lig != 7 || col != 7) {
                    res.append(",");
                }
            }
        }
        return res.toString();
    }

    @Override
    public String toString() {
        return damierReel.toString();
    }
    
     // Nouvelle méthode pour obtenir l'état du plateau en tant que tableau de doubles
    public double[] getBoardAsArray() {
        double[] boardArray = new double[64];
        int index = 0;
        for (int lig = 0; lig < 8; lig++) {
            for (int col = 0; col < 8; col++) {
                Case cur = this.damierReel.getVal(lig, col);
                if (cur == Case.VIDE) {
                    boardArray[index] = 0.0;
                } else if (cur == Case.NOIR) {
                    boardArray[index] = 1.0;
                } else if (cur == Case.BLANC) {
                    boardArray[index] = -1.0;
                }
                index++;
            }
        }
        return boardArray;
    }
    
    
    
}
