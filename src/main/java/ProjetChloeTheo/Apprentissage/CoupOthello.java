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
package ProjetChloeTheo.Apprentissage;

import ProjetChloeTheo.configuration_jeu_othello.Position;
import java.util.Optional;

/**
 * un coup à l'Othello : un ligne,colonne, ou passe.
 * <pre>
 * <p>
 * classe "proxy" pour la classe {@link ProjetChloeTheo.configuration_jeu_othello.Position}
 * </p>
 * </pre>
 *
 * @author francois
 */
public class CoupOthello {

    private Optional<int[]> pos;

    private CoupOthello() {
        this.pos = Optional.empty();
    }

    private CoupOthello(int lig, int col) {
        this.pos = Optional.of(new int[]{lig, col});
    }
    
    public static CoupOthello fromPos(Position p) {
        return new CoupOthello(p.getLigne(), p.getCol());
    }

    public static CoupOthello coupPasse() {
        return new CoupOthello();
    }

    public static CoupOthello coupNormal(int lig, int col) {
        return new CoupOthello(lig, col);
    }

    @Override
    public String toString() {
        if (this.getPos().isEmpty()) {
            return "[Passe]";
        } else {
            int[] p = this.getPos().get();
            return "[" + (char) ('A' + p[0]) + "," + (p[1]+1) + "]";
        }
    }

    public boolean isPasse() {
        return this.getPos().isEmpty();
    }

    /**
     * @return the pos
     */
    public Optional<int[]> getPos() {
        return pos;
    }
    
    public int getLig() {
        if (this.pos.isEmpty()) {
            throw new Error("passe : pas de ligne");
        }
        return this.pos.get()[0];
    }

    public int getCol() {
        if (this.pos.isEmpty()) {
            throw new Error("passe : pas de colonne");
        }
        return this.pos.get()[1];
    }

}
