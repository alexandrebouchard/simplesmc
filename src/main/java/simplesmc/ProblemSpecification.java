package simplesmc;


import org.apache.commons.lang3.tuple.Pair;

import bayonet.distributions.Random;


/**
 * The specification of a problem input to an SMC algorithm.
 * 
 * The specification provides:
 * - a way to propose a particle for the first iteration
 * - a way to propose a new particles given a particle from the
 *   previous iteration
 * - a weight update for the two operations above
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 * @param <P> The type of each individual particles 
 */
public interface ProblemSpecification<P>
{
  /**
   * Computes a proposal and the LOG weight update for that proposed particle.
   * 
   * @param currentSmcIteration The index of particle currentParticle (0, 1, 2, ..)
   * @param random
   * @param currentParticle
   * @return A pair of (1) LOG weight update, and (2) proposed particle
   */
  public Pair<Double, P>  proposeNext(int currentSmcIteration, Random random, P currentParticle);
  
  /**
   * 
   * @param random
   * @return A pair of (1) LOG weight update, and (2) proposed particle for the zeroth iteration
   */
  public Pair<Double, P>  proposeInitial(Random random);
  
  /**
   * @return Number of iterations, including the initial step. For example, this is the length of
   *   the chain in an HMM context
   */
  public int nIterations();
}
