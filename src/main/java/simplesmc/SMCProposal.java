package simplesmc;

import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;



public interface SMCProposal<P>
{
  public Pair<Double, P>  proposeNext(int previousSmcIteration, Random random, P currentParticle);
  
  public Pair<Double, P>  proposeInitial(Random random);
  
  /**
   * @return Number of iterations, including the initial step.
   */
  public int nIterations();
}
