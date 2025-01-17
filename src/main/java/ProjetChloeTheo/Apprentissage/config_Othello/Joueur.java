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

/**
 * Permet de désigner un des joueurs dans le jeu.
 * <p>
 * pour un jeu à deux joeurs, il suffit d'avoir deux valeurs possible
 * <p>
 * @author francois
 */
public enum Joueur {
    NOIR, BLANC;

    /**
     *
     * @return l'autre joueur
     */
    public Joueur adversaire() {
        if (this == NOIR) { //parfois noir est appelé J1
            return BLANC;   //parfois blanc est appelé J2
        } else {
            return NOIR;
        }
    }
}
