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

import ProjetChloeTheo.Apprentissage.oracles.OracleStupide;
import ProjetChloeTheo.Apprentissage.oracles.Oracle;
import ProjetChloeTheo.configuration_jeu_othello.Damier;
import ProjetChloeTheo.configuration_jeu_othello.Position;
import ProjetChloeTheo.utils.ListUtils;
import ProjetChloeTheo.utils.TiragesAlea2;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Une classe "proxy" pour donner les fonctionnalités d'un Jeu (au sens de l'api)
 * à une classe déjà existante {@link ProjetChloeTheo.configuration_jeu_othello.Damier}
 * qui implémente effectivement le jeu d'Othello.
 * @author francois
 */
public class JeuOthello {

    public SituationOthello situationInitiale() {
        Damier res = new Damier();
        return new SituationOthello(res);
    }

   /**
     * La représentation d'un joueur n'est pas la même dans le jeux original
     * et dans l'api.
     * <p> une petite conversion s'impose</p>
     * <p> notez ici l'utilisation du nom complet de la classe : je ne peux
     * pas faire simplement un import puisque j'utilise deux classe qui 
     * ont exactement le même nom.
     * </p>
     * @param j un Joueur au sens de l'api apprentissage
     * @return un Joueur au sens du jeu original d'othello
     */
    private ProjetChloeTheo.configuration_jeu_othello.Joueur convNewJoueurOldJoueur(Joueur j) {
        if (j == Joueur.NOIR) {
            return ProjetChloeTheo.configuration_jeu_othello.Joueur.NOIR;
        } else {
            return ProjetChloeTheo.configuration_jeu_othello.Joueur.BLANC;
        }
    }
    
    /**
     * La représentation d'un joueur n'est pas la même dans le jeux original
     * et dans l'api.
     * <p> une petite conversion s'impose</p>
     * <p> notez ici l'utilisation du nom complet de la classe : je ne peux
     * pas faire simplement un import puisque j'utilise deux classe qui 
     * ont exactement le même nom.
     * </p>
     * @param oldJ un Joueur au sens du jeu original d'othello
     * @return un Joueur au sens de l'api apprentissage 
     */
    private Joueur convOldJoueurNewJoueur(ProjetChloeTheo.configuration_jeu_othello.Joueur oldJ) {
        if (oldJ == ProjetChloeTheo.configuration_jeu_othello.Joueur.NOIR) {
            return Joueur.NOIR;
        } else {
            return Joueur.BLANC;
        }
    }

    /**
     * Test toutes les cases pour déterminier les coups jouables.
     * @param s
     * @param j
     * @return 
     */
    public List<CoupOthello> coupsJouables(SituationOthello s, Joueur j) {
        ProjetChloeTheo.configuration_jeu_othello.Joueur jConv
                = convNewJoueurOldJoueur(j);
        List<Position> jouables = s.getDamierReel().coupsJouables(jConv);
        if (jouables.isEmpty()) {
            return List.of(CoupOthello.coupPasse());
        } else {
            return jouables.stream().map(CoupOthello::fromPos).toList();
        }
    }

    public SituationOthello updateSituation(SituationOthello s, Joueur j, CoupOthello c) {
        if (c.isPasse()) {
            return s;
        } else {
            SituationOthello res = new SituationOthello(s.getDamierReel().copie());
            res.getDamierReel().effectueCoup(new Position(c.getLig(), c.getCol()), convNewJoueurOldJoueur(j));
            return res;
        }
    }

    public StatutSituation statutSituation(SituationOthello s) {
        Damier d = s.getDamierReel();
        if (d.auMoinsUneCaseJouable(convNewJoueurOldJoueur(Joueur.NOIR)) 
                || d.auMoinsUneCaseJouable(convNewJoueurOldJoueur(Joueur.BLANC))) {
            return StatutSituation.ENCOURS;
        } else {
            int pionsNoirs = d.comptePions(convNewJoueurOldJoueur(Joueur.NOIR));
            int pionsBlancs = d.comptePions(convNewJoueurOldJoueur(Joueur.BLANC));
            if (pionsNoirs > pionsBlancs) {
                return StatutSituation.NOIR_GAGNE;
            } else if (pionsBlancs > pionsNoirs) {
                return StatutSituation.BLANC_GAGNE;
            } else {
                return StatutSituation.MATCH_NUL;
            }
        }
    }
    
    public ResumeResultat partie(
            Oracle o1, ChoixCoup cc1,
            Oracle o2, ChoixCoup cc2,
            boolean j1Humain, boolean j2Humain,
            Random rand,
            boolean trace) {
        List<CoupOthello> res = new ArrayList<>();
        SituationOthello curSit = this.situationInitiale();
        List<Joueur> joueurs = List.of(Joueur.NOIR, Joueur.BLANC);
        List<Oracle> oracles = List.of(o1, o2);
        List<ChoixCoup> ccs = List.of(cc1, cc2);
        List<Boolean> humains = List.of(j1Humain, j2Humain);
        int numJoueur = 0;   //0 pour NOIR, 1 pour BLANC, donc o1 c'est tjr NOIR, o2 trj BLANC
        while (this.statutSituation(curSit) == StatutSituation.ENCOURS) {
            if (trace) {
                System.out.println("----- Sit actuelle -------");
                System.out.println(curSit);
            }
            
            Oracle curOracle = oracles.get(numJoueur);
            Oracle advOracle = oracles.get(1 - numJoueur);
            ChoixCoup curCC = ccs.get(numJoueur);
            Joueur curJoueur = joueurs.get(numJoueur);
            boolean humain = humains.get(numJoueur);
            List<CoupOthello> possibles = this.coupsJouables(curSit, curJoueur);
            CoupOthello coupChoisi;
            
            if (humain) {
                coupChoisi = ListUtils.selectOne("choisissez votre coup : ", possibles, Object::toString);
            } else {
                if (curCC == ChoixCoup.ALEA) {
                    // on ne tient aucun compte de l'oracle
                    // tirage aléatoire entre les coups possibles
                    int numCoup = rand.nextInt(possibles.size());  //editer un nombre aléatoire entre 0 et la taille de la liste de coups possibles
                    coupChoisi = possibles.get(numCoup);           //choisir le coup aléatoirement correspondant au numéro aléatoire numCoup
                } else {
                    
                    // je demande au stratege courant d'évaluer les coups jouables
                    // pour cela, je joue effectivement les coups, je calcule la
                    // nouvelle situation, et je demande à l'oracle ADVERSE de l'évaluer
                    List<Double> evals = new ArrayList<>();
                    for (CoupOthello coup : possibles) {       //pour chaque coup contenu sur chaque ligne de la liste possibles
                        SituationOthello nouvelle = this.updateSituation(curSit, curJoueur, coup); //
                        evals.add(advOracle.evalSituation(nouvelle));  //ajout de l'évaluation du pdv de l'adv 
                    }
                    if (curCC == ChoixCoup.ORACLE_MEILLEUR) {
                        // on prend le coup correspondant au MIN des évaluation
                        // car on a fait faire l'évaluation à l'adversaire
                        // je prend donc le coup qui amène à la situation la
                        // moins favorable pour l'adversaire
                        int imin = 0;
                        double min = evals.get(imin);
                        for (int i = 1; i < evals.size(); i++) {
                            if (evals.get(i) < min) {
                                imin = i;              // imin index du minimum en fin de boucle for
                                min = evals.get(imin); // minimum de la liste evals en fin de boucle for
                            }
                        }
                        coupChoisi = possibles.get(imin);  // les deux listes sont de même taille , donc je choisis le coup en index du coup le moins favorable pour adv
                    } else { // en fait on a curCC = ChoixCoup.ORACLE_PONDERE;
                        
                        // imaginons 4 coups possibles [c1,c2,c3,c4] évalués à evals = [0.2,0.7,0.6,0.1]
                        // !! il faut "inverser" puisque les évals sont du point de vue de l'adversaire
                        
                        List<Double> evalsInverse = evals.stream().map(x -> 1-x).toList(); // ça crée une list "evals" du pdv du curJoueur, donc ici on va chercehr le max pour meilleur coup
                        // ==> [0.8,0.3,0.4,0.9]
                        evalsInverse = evalsInverse.stream().map(x -> x*x). toList();
                        
                        // ensuite choix pondéré ==> il aura 2 fois plus de chance de choisir le premier coup que le troisième
                        // également 3 fois plus de chance de choisir le quatrieme coup que le deuxième
                        coupChoisi = TiragesAlea2.choixAleaPondere(possibles, evalsInverse, rand);
                    }
                }
            }
            if (trace) {
                System.out.println("Le joueur " + curJoueur + " joue en " + coupChoisi);
            }
            // je change la situation en fonction du coup réellement choisi
            curSit = this.updateSituation(curSit, curJoueur, coupChoisi);
            //je stocke le coup joué dans la liste de coups joués
            res.add(coupChoisi);
            // et je passe au joueur suivant sauf si passe
            numJoueur = 1 - numJoueur;
        }
        if (trace) {
            System.out.println("----- Sit finale -------");
            System.out.println(curSit);
            StatutSituation statut = this.statutSituation(curSit);
            if (statut == StatutSituation.MATCH_NUL) {
                System.out.println("Match nul");
            } else if (statut == StatutSituation.NOIR_GAGNE) {
                System.out.println("Gagnant : " + Joueur.NOIR);
            } else { //ou alors on a statut == StatutSituation.BLANC_GAGNE
                System.out.println("Gagnant : " + Joueur.BLANC);
            }
        }
        return new ResumeResultat(this.statutSituation(curSit), res);
    }

    /**
     * Organise nbrParties entre un oracle et un adversaire qui joue aléatoirement.
     * <p> 
     * @param oracle
     * @param nbrParties
     * @param r
     * @return le nombre de victoire, et le nombre de match nuls
     */
    public int[] partieVsAlea(Oracle oracle, int nbrParties, Random r) {
        Joueur jOracle = oracle.getEvalueSituationApresCoupDe();
        Oracle o1, o2;
        ChoixCoup cc1, cc2;
        if (jOracle == Joueur.NOIR) {
            o2 = new OracleStupide(Joueur.BLANC);
            o1 = oracle;
            cc1 = ChoixCoup.ALEA;
            cc2 = ChoixCoup.ORACLE_MEILLEUR;
        } else {
            o2 = oracle;
            o1 = new OracleStupide(Joueur.NOIR);
            cc1 = ChoixCoup.ORACLE_MEILLEUR;
            cc2 = ChoixCoup.ALEA;
        }
        int[] res = new int[2];
        for (int i = 0; i < nbrParties; i++) {
            ResumeResultat resOne = this.partie(o1, cc1, o2, cc2, false, false, r, false);
            if (resOne.getStatutFinal() == StatutSituation.MATCH_NUL) {
                res[1]++;
            } else if (resOne.getStatutFinal() == StatutSituation.NOIR_GAGNE && jOracle == Joueur.NOIR) {
                res[0]++;
            } else if (resOne.getStatutFinal() == StatutSituation.BLANC_GAGNE && jOracle == Joueur.BLANC) {
                res[0]++;
            }
        }
        return res;
    }

}
