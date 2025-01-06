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
package ProjetChloeTheo.Apprentissage.oracles;

import ProjetChloeTheo.Apprentissage.config_Othello.Joueur;
import ProjetChloeTheo.Apprentissage.config_Othello.SituationOthello;
import java.util.List;

/**
 *
 * @author francois
 */
public class OracleStupide implements Oracle{
    
    private Joueur evaluePour;
    
    public OracleStupide(Joueur evaluePour) {
        //super(List.of(Joueur.NOIR,Joueur.BLANC), evaluePour);
        this.evaluePour = evaluePour;
    }

    @Override
    public double evalSituation(SituationOthello s) {
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
